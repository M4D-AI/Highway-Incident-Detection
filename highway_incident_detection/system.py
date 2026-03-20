"""Runtime system for highway incident detection."""

import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from highway_incident_detection.audit.event_auditor import EventAuditor
from highway_incident_detection.config.settings import Settings, get_settings
from highway_incident_detection.detectors.event_detector import (
    EventDetectionContext,
    EventDetector,
    EventDetectorHandler,
    RegisteredEventDetector,
)
from highway_incident_detection.detectors.loader import register_detectors_from_config
from highway_incident_detection.detectors.event_types import EventTypeLike
from highway_incident_detection.detectors.event_types import IncidentEvent
from highway_incident_detection.models.yolo_model import YOLOModel
from highway_incident_detection.output.result_output import ResultHandler
from highway_incident_detection.stream.rtsp_stream import RTSPStreamProcessor
from highway_incident_detection.summary.summary_generator import SummaryGenerator
from highway_incident_detection.utils.logger import get_logger, setup_logger
from highway_incident_detection.utils.resource_monitor import ResourceMonitor
from highway_incident_detection.utils.roi_filter import filter_detections_by_roi
from highway_incident_detection.utils.tracker import ObjectTracker


class HighwayIncidentDetectionSystem:
    """Main system class for highway incident detection."""

    def __init__(self, config_path: str = None):
        """Initialize the system."""
        if config_path:
            self.settings = Settings.from_yaml(config_path)
        else:
            self.settings = get_settings()

        setup_logger(
            log_dir=self.settings.logging.log_dir,
            log_level=self.settings.logging.log_level,
            log_rotation=self.settings.logging.log_rotation,
            log_retention=self.settings.logging.log_retention,
            log_format=self.settings.logging.log_format,
        )
        self.logger = get_logger(__name__)

        self.yolo_model = YOLOModel(self.settings.yolo)
        self.event_detector = EventDetector(self.settings.event)
        self._load_configured_event_detectors()
        self.auditor = EventAuditor(self.settings.audit)
        self.summary_generator = SummaryGenerator(
            self.settings.output,
            auditor=self.auditor,
        )

        self.trackers: Dict[str, ObjectTracker] = {}
        self.lane_configs: Dict[str, Dict] = {}
        self._init_lane_configs()
        self._inference_lock = threading.Lock()
        self._event_detector_lock = threading.Lock()
        self._stream_locks: Dict[str, threading.Lock] = {
            stream_config.get("name"): threading.Lock()
            for stream_config in self.settings.stream.streams
            if stream_config.get("name")
        }

        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.stream_processor = RTSPStreamProcessor(
            self.settings.stream,
            frame_callback=self._process_frame,
        )
        self.resource_monitor = ResourceMonitor(
            memory_threshold_mb=4096.0,
            memory_threshold_percent=80.0,
            cpu_threshold_percent=90.0,
        )

        self.running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event: Optional[asyncio.Event] = None

    def _load_configured_event_detectors(self) -> None:
        """Load custom detectors declared in configuration."""
        try:
            registered_names = register_detectors_from_config(
                self.event_detector,
                self.settings.event.custom_detectors,
            )
        except Exception as e:
            self.logger.error(f"Failed to load configured custom detectors: {e}")
            raise

        if registered_names:
            self.logger.info(
                f"Loaded custom event detectors from config: {', '.join(registered_names)}"
            )

    def _init_lane_configs(self) -> None:
        """Initialize lane configurations for each stream."""
        for stream_config in self.settings.stream.streams:
            stream_name = stream_config.get("name")
            lane_config = stream_config.get("lane", {})
            if lane_config:
                self.lane_configs[stream_name] = {
                    "enabled": lane_config.get("enabled", False),
                    "roi_points": lane_config.get("roi_points", []),
                    "direction": lane_config.get("direction"),
                }
            else:
                self.lane_configs[stream_name] = {
                    "enabled": self.settings.lane.enabled,
                    "roi_points": self.settings.lane.roi_points,
                    "direction": self.settings.lane.direction,
                }

    def _analyze_frame(
        self, stream_name: str, frame: np.ndarray, frame_number: int, timestamp: datetime
    ) -> List[IncidentEvent]:
        """Run the synchronous detection pipeline for a single frame."""
        stream_lock = self._stream_locks.setdefault(stream_name, threading.Lock())
        with stream_lock:
            with self._inference_lock:
                detections = self.yolo_model.predict(frame, verbose=False)

            lane_config = self.lane_configs.get(stream_name, {})
            if lane_config.get("enabled", False):
                roi_points = lane_config.get("roi_points", [])
                if roi_points:
                    detections = filter_detections_by_roi(detections, roi_points)

            if stream_name not in self.trackers:
                self.trackers[stream_name] = ObjectTracker(
                    max_disappeared=5,
                    max_distance=50.0,
                )

            tracker_detections = [(class_id, bbox) for class_id, _, bbox in detections]
            tracked_objects = self.trackers[stream_name].update(
                tracker_detections,
                frame_number,
            )

            with self._event_detector_lock:
                events = self.event_detector.process_frame(
                    detections=detections,
                    tracked_objects=tracked_objects,
                    stream_name=stream_name,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    frame_width=frame.shape[1] if frame is not None else None,
                    frame_height=frame.shape[0] if frame is not None else None,
                )

            self.summary_generator.add_frame(stream_name, frame, frame_number, timestamp)
        return events

    async def _process_frame(
        self, stream_name: str, frame: np.ndarray, frame_number: int, timestamp: datetime
    ) -> None:
        """Process a single frame without blocking the event loop."""
        try:
            events = await asyncio.to_thread(
                self._analyze_frame,
                stream_name,
                frame,
                frame_number,
                timestamp,
            )
            for event in events:
                self.logger.info(
                    f"Event detected: {event.event_type_value} "
                    f"on {event.stream_name} at frame {event.frame_number} "
                    f"(confidence: {event.confidence:.2f})"
                )
                await self.event_queue.put(event)
        except Exception as e:
            self.logger.error(f"Error processing frame from {stream_name}: {e}")

    async def _process_event_queue(self) -> None:
        """Process events from queue and forward to summary generator."""
        while self.running or not self.event_queue.empty():
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                try:
                    await self.summary_generator.add_event(event)
                finally:
                    self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event queue: {e}")

    async def _monitor_resources(self) -> None:
        """Monitor system resources and perform health checks."""
        while self.running:
            try:
                await asyncio.sleep(60)

                is_healthy, message = self.resource_monitor.check_health()
                if not is_healthy:
                    self.logger.warning(f"Resource health check failed: {message}")
                    self.resource_monitor.force_gc()

                stats = self.resource_monitor.get_stats()
                self.logger.debug(
                    f"Resource stats: Memory={stats.memory_mb:.1f}MB "
                    f"({stats.memory_percent:.1f}%), CPU={stats.cpu_percent:.1f}%"
                )

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
        self._shutdown_event = asyncio.Event()

        await self.summary_generator.start()

        event_task = asyncio.create_task(self._process_event_queue())
        self._monitoring_task = asyncio.create_task(self._monitor_resources())
        stream_task = asyncio.create_task(self.stream_processor.start())

        try:
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())
            _, pending = await asyncio.wait(
                [stream_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        finally:
            self.running = False
            if self._shutdown_event:
                self._shutdown_event.set()

            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

            if not stream_task.done():
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass

            await self.shutdown()

            if not event_task.done():
                await event_task

    async def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        self.logger.info("Shutting down system...")
        self.running = False

        try:
            await asyncio.wait_for(self.stream_processor.stop(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Stream processor stop timeout, forcing shutdown")

        try:
            await asyncio.wait_for(self.event_queue.join(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Event forwarding queue drain timeout")

        try:
            await asyncio.wait_for(self.summary_generator.stop(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Summary generator stop timeout, forcing shutdown")

        self.logger.info("System shutdown complete")

    def register_result_handler(self, handler: ResultHandler) -> None:
        """Register a downstream result handler."""
        self.summary_generator.add_result_handler(handler)

    def unregister_result_handler(self, handler: ResultHandler) -> None:
        """Unregister a downstream result handler."""
        self.summary_generator.remove_result_handler(handler)

    def register_event_detector(
        self,
        name: str,
        event_type: EventTypeLike,
        handler: EventDetectorHandler,
        *,
        prepend: bool = False,
    ) -> None:
        """Register a custom event detector."""
        self.event_detector.register_detector(
            name=name,
            event_type=event_type,
            handler=handler,
            prepend=prepend,
        )

    def unregister_event_detector(self, name: str) -> None:
        """Unregister a custom or built-in event detector by name."""
        self.event_detector.unregister_detector(name)

    def list_event_detectors(self) -> List[RegisteredEventDetector]:
        """Return registered event detectors in execution order."""
        return self.event_detector.list_detectors()
