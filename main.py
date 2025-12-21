"""Main entry point for Highway Incident Detection System."""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from datetime import datetime

from highway_incident_detection.config.settings import Settings, get_settings
from highway_incident_detection.models.yolo_model import YOLOModel
from highway_incident_detection.detectors.event_detector import EventDetector
from highway_incident_detection.detectors.event_types import IncidentEvent
from highway_incident_detection.stream.rtsp_stream import RTSPStreamProcessor
from highway_incident_detection.summary.summary_generator import SummaryGenerator
from highway_incident_detection.utils.logger import setup_logger, get_logger
from highway_incident_detection.utils.tracker import ObjectTracker
from highway_incident_detection.utils.resource_monitor import ResourceMonitor


class HighwayIncidentDetectionSystem:
    """Main system class for highway incident detection."""

    def __init__(self, config_path: str = None):
        """Initialize the system.

        Args:
            config_path: Optional path to YAML configuration file.
        """
        # Load settings
        if config_path:
            self.settings = Settings.from_yaml(config_path)
        else:
            self.settings = get_settings()

        # Setup logger
        setup_logger(
            log_dir=self.settings.logging.log_dir,
            log_level=self.settings.logging.log_level,
            log_rotation=self.settings.logging.log_rotation,
            log_retention=self.settings.logging.log_retention,
            log_format=self.settings.logging.log_format,
        )
        self.logger = get_logger(__name__)

        # Initialize components
        self.yolo_model = YOLOModel(self.settings.yolo)
        self.event_detector = EventDetector(self.settings.event)
        self.summary_generator = SummaryGenerator(self.settings.output)

        # Per-stream trackers
        self.trackers: Dict[str, ObjectTracker] = {}

        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue()

        # Stream processor
        self.stream_processor = RTSPStreamProcessor(
            self.settings.stream,
            frame_callback=self._process_frame,
        )

        # Resource monitoring
        self.resource_monitor = ResourceMonitor(
            memory_threshold_mb=4096.0,
            memory_threshold_percent=80.0,
            cpu_threshold_percent=90.0,
        )

        self.running = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._monitoring_task: Optional[asyncio.Task] = None

    def _process_frame(
        self, stream_name: str, frame: np.ndarray, frame_number: int, timestamp: datetime
    ) -> None:
        """Process a single frame (called by stream processor).

        Args:
            stream_name: Name of the stream.
            frame: Frame image.
            frame_number: Frame number.
            timestamp: Frame timestamp.
        """
        try:
            # Initialize tracker for stream if needed
            if stream_name not in self.trackers:
                self.trackers[stream_name] = ObjectTracker(
                    max_disappeared=5,
                    max_distance=50.0,
                )

            # Run YOLO inference
            detections = self.yolo_model.predict(frame, verbose=False)

            # Convert detections to format for tracker
            tracker_detections = [
                (class_id, bbox) for class_id, _, bbox in detections
            ]

            # Update tracker
            tracked_objects = self.trackers[stream_name].update(
                tracker_detections, frame_number
            )

            # Detect events
            events = self.event_detector.process_frame(
                detections=[(class_id, conf, bbox) for class_id, conf, bbox in detections],
                tracked_objects=tracked_objects,
                stream_name=stream_name,
                frame_number=frame_number,
                timestamp=timestamp,
            )

            # Add frame to summary generator buffer
            self.summary_generator.add_frame(
                stream_name, frame, frame_number, timestamp
            )

            # Process events asynchronously (non-blocking)
            for event in events:
                self.logger.info(
                    f"Event detected: {event.event_type.value} "
                    f"on {event.stream_name} at frame {event.frame_number} "
                    f"(confidence: {event.confidence:.2f})"
                )
                # Add to async queue (non-blocking)
                if self._event_loop and self._event_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.summary_generator.add_event(event),
                        self._event_loop
                    )
                else:
                    # Fallback: add to queue directly if loop not available
                    try:
                        self.event_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        self.logger.warning("Event queue full, dropping event")

        except Exception as e:
            self.logger.error(f"Error processing frame from {stream_name}: {e}")

    async def _process_event_queue(self) -> None:
        """Process events from queue and forward to summary generator."""
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self.summary_generator.add_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event queue: {e}")

    async def _monitor_resources(self) -> None:
        """Monitor system resources and perform health checks."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Check resource health
                is_healthy, message = self.resource_monitor.check_health()
                if not is_healthy:
                    self.logger.warning(f"Resource health check failed: {message}")
                    # Force garbage collection
                    self.resource_monitor.force_gc()

                # Log resource stats periodically
                stats = self.resource_monitor.get_stats()
                self.logger.debug(
                    f"Resource stats: Memory={stats.memory_mb:.1f}MB "
                    f"({stats.memory_percent:.1f}%), CPU={stats.cpu_percent:.1f}%"
                )

                # Log model stats
                model_stats = self.yolo_model.get_stats()
                if model_stats["total_inferences"] > 0:
                    failure_rate = (
                        model_stats["total_failures"] / model_stats["total_inferences"] * 100
                    )
                    self.logger.debug(
                        f"Model stats: Inferences={model_stats['total_inferences']}, "
                        f"Failures={model_stats['total_failures']} "
                        f"({failure_rate:.2f}%), "
                        f"Consecutive failures={model_stats['consecutive_failures']}"
                    )

                # Log stream status
                stream_status = self.stream_processor.get_stream_status()
                for stream_name, status in stream_status.items():
                    if not status["is_connected"]:
                        self.logger.warning(
                            f"Stream {stream_name} is not connected "
                            f"(reconnect attempts: {status['reconnect_attempts']})"
                        )

            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")

    async def run(self) -> None:
        """Run the system."""
        self.logger.info("Starting Highway Incident Detection System...")
        self.running = True
        self._event_loop = asyncio.get_event_loop()

        # Start summary generator
        await self.summary_generator.start()

        # Start event queue processor
        event_task = asyncio.create_task(self._process_event_queue())

        # Start resource monitoring
        self._monitoring_task = asyncio.create_task(self._monitor_resources())

        # Start stream processing
        try:
            await self.stream_processor.start()
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        finally:
            # Cancel monitoring task
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

            # Cancel event task
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass

            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        self.logger.info("Shutting down system...")
        self.running = False

        # Stop stream processor
        await self.stream_processor.stop()

        # Stop summary generator
        await self.summary_generator.stop()

        self.logger.info("System shutdown complete")


# Main entry point is now in highway_incident_detection.cli
# This file is kept for backward compatibility
if __name__ == "__main__":
    from highway_incident_detection.cli import main
    main()

