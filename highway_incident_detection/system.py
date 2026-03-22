"""Runtime system for highway incident detection."""

import asyncio
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class FrameEnvelope:
    """Frame data exchanged between stream readers and inference workers."""

    stream_name: str
    frame: np.ndarray
    frame_number: int
    timestamp: datetime
    enqueued_at_monotonic: float = 0.0
    queue_depth_at_enqueue: int = 0


@dataclass
class FrameInferenceResult:
    """Inference result for one frame after micro-batch detection."""

    envelope: FrameEnvelope
    detections: List[Tuple[int, float, Tuple[float, float, float, float]]]
    batch_size: int
    batch_timed_out: bool
    queue_wait_ms: float
    inference_duration_ms: float
    ready_at_monotonic: float


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
        self.stream_event_filters: Dict[str, Dict[str, Optional[set[str]]]] = {}
        self._init_lane_configs()
        self._init_stream_event_filters()
        self._inference_lock = threading.Lock()
        self._event_detector_lock = threading.Lock()
        self._stream_locks: Dict[str, threading.Lock] = {
            stream_config.get("name"): threading.Lock()
            for stream_config in self.settings.stream.streams
            if stream_config.get("name")
        }
        self._pending_inference_results: Dict[str, Dict[int, FrameInferenceResult]] = {}
        self._next_frame_to_process: Dict[str, int] = {}
        self._batch_metrics_lock = threading.Lock()
        self._batch_metrics = self._empty_batch_metrics_state()

        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.inference_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.settings.yolo.batch_queue_size
        )
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
        self._inference_task: Optional[asyncio.Task] = None

    @staticmethod
    def _empty_batch_metrics_state() -> Dict[str, Any]:
        """Create the mutable batch metrics state container."""
        return {
            "total_enqueued_frames": 0,
            "total_completed_frames": 0,
            "total_batches": 0,
            "total_frames_in_batches": 0,
            "timeout_batches": 0,
            "full_batches": 0,
            "max_batch_size_seen": 0,
            "max_queue_depth_seen": 0,
            "total_queue_wait_ms": 0.0,
            "total_inference_duration_ms": 0.0,
            "total_postprocess_wait_ms": 0.0,
            "per_stream": {},
        }

    @staticmethod
    def _empty_stream_batch_metrics() -> Dict[str, Any]:
        """Create the mutable per-stream metrics state container."""
        return {
            "enqueued_frames": 0,
            "completed_frames": 0,
            "last_frame_number": 0,
            "last_queue_wait_ms": 0.0,
            "last_postprocess_wait_ms": 0.0,
            "total_queue_wait_ms": 0.0,
            "total_postprocess_wait_ms": 0.0,
        }

    def _record_batch_enqueue(self, stream_name: str, queue_depth: int) -> None:
        """Record queueing metrics for one frame."""
        with self._batch_metrics_lock:
            metrics = self._batch_metrics
            metrics["total_enqueued_frames"] += 1
            metrics["max_queue_depth_seen"] = max(metrics["max_queue_depth_seen"], queue_depth)
            stream_metrics = metrics["per_stream"].setdefault(
                stream_name,
                self._empty_stream_batch_metrics(),
            )
            stream_metrics["enqueued_frames"] += 1

    def _record_batch_completion(
        self,
        batch: List[FrameEnvelope],
        *,
        batch_timed_out: bool,
        inference_duration_ms: float,
        queue_waits_ms: List[float],
    ) -> None:
        """Record one completed inference batch."""
        with self._batch_metrics_lock:
            metrics = self._batch_metrics
            batch_size = len(batch)
            metrics["total_batches"] += 1
            metrics["total_frames_in_batches"] += batch_size
            metrics["total_completed_frames"] += batch_size
            metrics["max_batch_size_seen"] = max(metrics["max_batch_size_seen"], batch_size)
            metrics["total_inference_duration_ms"] += inference_duration_ms
            if batch_timed_out:
                metrics["timeout_batches"] += 1
            if batch_size >= self.settings.yolo.max_batch_size:
                metrics["full_batches"] += 1

            for envelope, queue_wait_ms in zip(batch, queue_waits_ms):
                metrics["total_queue_wait_ms"] += queue_wait_ms
                stream_metrics = metrics["per_stream"].setdefault(
                    envelope.stream_name,
                    self._empty_stream_batch_metrics(),
                )
                stream_metrics["completed_frames"] += 1
                stream_metrics["last_frame_number"] = envelope.frame_number
                stream_metrics["last_queue_wait_ms"] = queue_wait_ms
                stream_metrics["total_queue_wait_ms"] += queue_wait_ms

    def _record_postprocess_wait(
        self,
        stream_name: str,
        frame_number: int,
        postprocess_wait_ms: float,
    ) -> None:
        """Record post-inference waiting time before ordered tracker/event processing."""
        with self._batch_metrics_lock:
            metrics = self._batch_metrics
            metrics["total_postprocess_wait_ms"] += postprocess_wait_ms
            stream_metrics = metrics["per_stream"].setdefault(
                stream_name,
                self._empty_stream_batch_metrics(),
            )
            stream_metrics["last_frame_number"] = frame_number
            stream_metrics["last_postprocess_wait_ms"] = postprocess_wait_ms
            stream_metrics["total_postprocess_wait_ms"] += postprocess_wait_ms

    def _attach_processing_metrics(
        self,
        events: List[IncidentEvent],
        inference_result: FrameInferenceResult,
        *,
        postprocess_wait_ms: float,
    ) -> None:
        """Attach micro-batch processing metrics to emitted events."""
        processing_metrics = {
            "mode": "micro_batch",
            "batch_size": inference_result.batch_size,
            "batch_timed_out": inference_result.batch_timed_out,
            "queue_wait_ms": round(inference_result.queue_wait_ms, 3),
            "postprocess_wait_ms": round(postprocess_wait_ms, 3),
            "inference_duration_ms": round(inference_result.inference_duration_ms, 3),
            "queue_depth_at_enqueue": inference_result.envelope.queue_depth_at_enqueue,
        }
        for event in events:
            event.metadata = dict(event.metadata or {})
            event.metadata["processing_metrics"] = processing_metrics.copy()

    def get_batch_inference_stats(self) -> Dict[str, Any]:
        """Return a snapshot of micro-batch runtime metrics."""
        with self._batch_metrics_lock:
            metrics = self._batch_metrics
            total_frames = metrics["total_completed_frames"]
            total_batches = metrics["total_batches"]
            per_stream = {}
            for stream_name, stream_metrics in metrics["per_stream"].items():
                completed_frames = stream_metrics["completed_frames"]
                per_stream[stream_name] = {
                    "enqueued_frames": stream_metrics["enqueued_frames"],
                    "completed_frames": completed_frames,
                    "last_frame_number": stream_metrics["last_frame_number"],
                    "last_queue_wait_ms": round(stream_metrics["last_queue_wait_ms"], 3),
                    "last_postprocess_wait_ms": round(
                        stream_metrics["last_postprocess_wait_ms"], 3
                    ),
                    "avg_queue_wait_ms": round(
                        stream_metrics["total_queue_wait_ms"] / completed_frames, 3
                    )
                    if completed_frames
                    else 0.0,
                    "avg_postprocess_wait_ms": round(
                        stream_metrics["total_postprocess_wait_ms"] / completed_frames, 3
                    )
                    if completed_frames
                    else 0.0,
                }

            return {
                "enabled": self.settings.yolo.batch_inference_enabled,
                "queue_size": self.inference_queue.qsize(),
                "queue_capacity": self.settings.yolo.batch_queue_size,
                "total_enqueued_frames": metrics["total_enqueued_frames"],
                "total_completed_frames": total_frames,
                "total_batches": total_batches,
                "timeout_batches": metrics["timeout_batches"],
                "full_batches": metrics["full_batches"],
                "timeout_rate": round(metrics["timeout_batches"] / total_batches, 4)
                if total_batches
                else 0.0,
                "avg_batch_size": round(metrics["total_frames_in_batches"] / total_batches, 3)
                if total_batches
                else 0.0,
                "max_batch_size_seen": metrics["max_batch_size_seen"],
                "max_queue_depth_seen": metrics["max_queue_depth_seen"],
                "avg_queue_wait_ms": round(metrics["total_queue_wait_ms"] / total_frames, 3)
                if total_frames
                else 0.0,
                "avg_postprocess_wait_ms": round(
                    metrics["total_postprocess_wait_ms"] / total_frames, 3
                )
                if total_frames
                else 0.0,
                "avg_inference_duration_ms": round(
                    metrics["total_inference_duration_ms"] / total_batches, 3
                )
                if total_batches
                else 0.0,
                "per_stream": per_stream,
            }

    def get_runtime_stats(self) -> Dict[str, Any]:
        """Return a consolidated runtime snapshot for external inspection."""
        return {
            "model": self.yolo_model.get_stats(),
            "streams": self.stream_processor.get_stream_status(),
            "batch_inference": self.get_batch_inference_stats(),
        }

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
                self.lane_configs[stream_name] = self._normalize_lane_config(lane_config)
            else:
                self.lane_configs[stream_name] = self._normalize_lane_config(
                    self.settings.lane.model_dump()
                )

    def _init_stream_event_filters(self) -> None:
        """Initialize per-stream event filters."""
        self.stream_event_filters = {}
        for stream_config in self.settings.stream.streams:
            stream_name = stream_config.get("name")
            if not stream_name:
                continue
            enabled = stream_config.get("enabled_event_types")
            disabled = stream_config.get("disabled_event_types")
            self.stream_event_filters[stream_name] = {
                "enabled": {str(item) for item in enabled} if enabled else None,
                "disabled": {str(item) for item in disabled} if disabled else None,
            }

    @staticmethod
    def _normalize_lane_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize single-lane and multi-lane configs to one internal shape."""
        enabled = raw_config.get("enabled", False)
        zones = []
        for index, zone in enumerate(raw_config.get("zones", [])):
            zones.append(
                {
                    "name": zone.get("name", f"lane_{index + 1}"),
                    "roi_points": zone.get("roi_points", []),
                    "direction": zone.get("direction"),
                }
            )

        if enabled and not zones:
            zones.append(
                {
                    "name": "default",
                    "roi_points": raw_config.get("roi_points", []),
                    "direction": raw_config.get("direction"),
                }
            )

        return {
            "enabled": enabled,
            "zones": zones,
        }

    @staticmethod
    def _zone_state_key(stream_name: str, zone_name: str) -> str:
        """Build the internal state key for one lane zone."""
        return f"{stream_name}::{zone_name}"

    def _process_zone(
        self,
        stream_name: str,
        zone_name: str,
        zone_direction: Optional[str],
        enabled_event_types: Optional[set[str]],
        disabled_event_types: Optional[set[str]],
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
        frame: np.ndarray,
        frame_number: int,
        timestamp: datetime,
    ) -> List[IncidentEvent]:
        """Run tracking and event detection for one zone."""
        zone_state_key = self._zone_state_key(stream_name, zone_name)
        if zone_state_key not in self.trackers:
            self.trackers[zone_state_key] = ObjectTracker(
                max_disappeared=5,
                max_distance=50.0,
            )

        tracker_detections = [(class_id, bbox) for class_id, _, bbox in detections]
        tracked_objects = self.trackers[zone_state_key].update(
            tracker_detections,
            frame_number,
        )

        events = self.event_detector.process_frame(
            detections=detections,
            tracked_objects=tracked_objects,
            stream_name=zone_state_key,
            frame_number=frame_number,
            timestamp=timestamp,
            frame_width=frame.shape[1] if frame is not None else None,
            frame_height=frame.shape[0] if frame is not None else None,
            lane_direction=zone_direction,
            enabled_event_types=enabled_event_types,
            disabled_event_types=disabled_event_types,
        )

        for event in events:
            event.stream_name = stream_name
            event.metadata["lane_zone"] = zone_name
            event.metadata["lane_direction"] = zone_direction

        return events

    def _analyze_frame(
        self, stream_name: str, frame: np.ndarray, frame_number: int, timestamp: datetime
    ) -> List[IncidentEvent]:
        """Run the synchronous detection pipeline for a single frame."""
        stream_lock = self._stream_locks.setdefault(stream_name, threading.Lock())
        with stream_lock:
            with self._inference_lock:
                detections = self.yolo_model.predict(frame, verbose=False)
            return self._finalize_frame_locked(
                stream_name=stream_name,
                frame=frame,
                frame_number=frame_number,
                timestamp=timestamp,
                detections=detections,
            )

    def _finalize_frame_locked(
        self,
        stream_name: str,
        frame: np.ndarray,
        frame_number: int,
        timestamp: datetime,
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
    ) -> List[IncidentEvent]:
        """Run post-inference per-stream processing in frame order."""
        lane_config = self.lane_configs.get(stream_name, {})
        event_filter = self.stream_event_filters.get(
            stream_name,
            {"enabled": None, "disabled": None},
        )
        with self._event_detector_lock:
            if lane_config.get("enabled", False) and lane_config.get("zones"):
                events: List[IncidentEvent] = []
                for zone in lane_config["zones"]:
                    zone_detections = detections
                    if zone.get("roi_points"):
                        zone_detections = filter_detections_by_roi(
                            detections,
                            zone["roi_points"],
                        )
                    events.extend(
                        self._process_zone(
                            stream_name=stream_name,
                            zone_name=zone["name"],
                            zone_direction=zone.get("direction"),
                            enabled_event_types=event_filter["enabled"],
                            disabled_event_types=event_filter["disabled"],
                            detections=zone_detections,
                            frame=frame,
                            frame_number=frame_number,
                            timestamp=timestamp,
                        )
                    )
            else:
                events = self._process_zone(
                    stream_name=stream_name,
                    zone_name="default",
                    zone_direction=None,
                    enabled_event_types=event_filter["enabled"],
                    disabled_event_types=event_filter["disabled"],
                    detections=detections,
                    frame=frame,
                    frame_number=frame_number,
                    timestamp=timestamp,
                )

        self.summary_generator.add_frame(stream_name, frame, frame_number, timestamp)
        return events

    async def _process_detected_events(self, events: List[IncidentEvent]) -> None:
        """Forward detected events to the async event queue."""
        for event in events:
            self.logger.info(
                f"Event detected: {event.event_type_value} "
                f"on {event.stream_name} at frame {event.frame_number} "
                f"(confidence: {event.confidence:.2f})"
            )
            await self.event_queue.put(event)

    async def _enqueue_frame_for_batch_inference(
        self,
        stream_name: str,
        frame: np.ndarray,
        frame_number: int,
        timestamp: datetime,
    ) -> None:
        """Queue one frame for micro-batch inference."""
        loop = asyncio.get_running_loop()
        await self.inference_queue.put(
            FrameEnvelope(
                stream_name=stream_name,
                frame=frame,
                frame_number=frame_number,
                timestamp=timestamp,
                enqueued_at_monotonic=loop.time(),
                queue_depth_at_enqueue=self.inference_queue.qsize(),
            )
        )
        self._record_batch_enqueue(stream_name, self.inference_queue.qsize())

    async def _collect_inference_batch(self) -> Tuple[List[FrameEnvelope], bool]:
        """Collect a micro-batch of frames for one inference call."""
        batch: List[FrameEnvelope] = []
        batch_timed_out = False
        first_item = await asyncio.wait_for(self.inference_queue.get(), timeout=1.0)
        batch.append(first_item)

        deadline = asyncio.get_running_loop().time() + (
            self.settings.yolo.batch_timeout_ms / 1000.0
        )
        while len(batch) < self.settings.yolo.max_batch_size:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                batch_timed_out = True
                break
            try:
                next_item = await asyncio.wait_for(self.inference_queue.get(), timeout=remaining)
                batch.append(next_item)
            except asyncio.TimeoutError:
                batch_timed_out = True
                break
        return batch, batch_timed_out

    async def _run_inference_batch(
        self,
        batch: List[FrameEnvelope],
    ) -> List[List[Tuple[int, float, Tuple[float, float, float, float]]]]:
        """Run one inference batch using the shared YOLO model."""
        if not batch:
            return []

        images = [item.frame for item in batch]
        with self._inference_lock:
            if len(images) == 1:
                return [self.yolo_model.predict(images[0], verbose=False)]
            detections_by_frame = self.yolo_model.predict_batch(images, verbose=False)

        if len(detections_by_frame) != len(batch):
            self.logger.warning(
                "Batch inference returned mismatched result count: "
                f"expected={len(batch)}, actual={len(detections_by_frame)}. "
                "Missing results will be filled with empty detections."
            )
            detections_by_frame = list(detections_by_frame[: len(batch)])
            if len(detections_by_frame) < len(batch):
                detections_by_frame.extend([[] for _ in range(len(batch) - len(detections_by_frame))])

        return detections_by_frame

    async def _drain_ready_stream_results(self, stream_name: str) -> None:
        """Process ready inference results for one stream in strict frame order."""
        stream_lock = self._stream_locks.setdefault(stream_name, threading.Lock())
        pending_results = self._pending_inference_results.setdefault(stream_name, {})
        next_frame = self._next_frame_to_process.setdefault(stream_name, 1)
        loop = asyncio.get_running_loop()

        while next_frame in pending_results:
            inference_result = pending_results.pop(next_frame)
            postprocess_wait_ms = max(
                0.0,
                (loop.time() - inference_result.ready_at_monotonic) * 1000.0,
            )
            self._record_postprocess_wait(
                inference_result.envelope.stream_name,
                inference_result.envelope.frame_number,
                postprocess_wait_ms,
            )
            events = await asyncio.to_thread(
                self._finalize_inference_result,
                stream_lock,
                inference_result.envelope,
                inference_result.detections,
            )
            self._attach_processing_metrics(
                events,
                inference_result,
                postprocess_wait_ms=postprocess_wait_ms,
            )
            await self._process_detected_events(events)
            next_frame += 1

        self._next_frame_to_process[stream_name] = next_frame

    def _finalize_inference_result(
        self,
        stream_lock: threading.Lock,
        envelope: FrameEnvelope,
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
    ) -> List[IncidentEvent]:
        """Finalize one frame after batched inference."""
        with stream_lock:
            return self._finalize_frame_locked(
                stream_name=envelope.stream_name,
                frame=envelope.frame,
                frame_number=envelope.frame_number,
                timestamp=envelope.timestamp,
                detections=detections,
            )

    async def _inference_batch_loop(self) -> None:
        """Run micro-batch inference and dispatch results back to each stream."""
        while self.running or not self.inference_queue.empty():
            try:
                batch, batch_timed_out = await self._collect_inference_batch()
            except asyncio.TimeoutError:
                continue

            try:
                loop = asyncio.get_running_loop()
                inference_started = loop.time()
                detections_by_frame = await self._run_inference_batch(batch)
                inference_finished = loop.time()
                inference_duration_ms = max(
                    0.0,
                    (inference_finished - inference_started) * 1000.0,
                )
                queue_waits_ms = [
                    max(0.0, (inference_started - envelope.enqueued_at_monotonic) * 1000.0)
                    for envelope in batch
                ]
                self._record_batch_completion(
                    batch,
                    batch_timed_out=batch_timed_out,
                    inference_duration_ms=inference_duration_ms,
                    queue_waits_ms=queue_waits_ms,
                )
                touched_streams = set()
                for envelope, detections, queue_wait_ms in zip(
                    batch,
                    detections_by_frame,
                    queue_waits_ms,
                ):
                    pending_results = self._pending_inference_results.setdefault(
                        envelope.stream_name,
                        {},
                    )
                    pending_results[envelope.frame_number] = FrameInferenceResult(
                        envelope=envelope,
                        detections=detections,
                        batch_size=len(batch),
                        batch_timed_out=batch_timed_out,
                        queue_wait_ms=queue_wait_ms,
                        inference_duration_ms=inference_duration_ms,
                        ready_at_monotonic=inference_finished,
                    )
                    touched_streams.add(envelope.stream_name)

                for stream_name in touched_streams:
                    await self._drain_ready_stream_results(stream_name)
            except Exception as e:
                self.logger.error(f"Error in inference batch loop: {e}")
            finally:
                for _ in batch:
                    self.inference_queue.task_done()

    async def _process_frame(
        self, stream_name: str, frame: np.ndarray, frame_number: int, timestamp: datetime
    ) -> None:
        """Process a single frame without blocking the event loop."""
        try:
            if self.settings.yolo.batch_inference_enabled:
                await self._enqueue_frame_for_batch_inference(
                    stream_name,
                    frame,
                    frame_number,
                    timestamp,
                )
                return

            events = await asyncio.to_thread(
                self._analyze_frame,
                stream_name,
                frame,
                frame_number,
                timestamp,
            )
            await self._process_detected_events(events)
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

                if self.settings.yolo.batch_inference_enabled:
                    batch_stats = self.get_batch_inference_stats()
                    self.logger.debug(
                        "Batch stats: "
                        f"Batches={batch_stats['total_batches']}, "
                        f"Avg batch={batch_stats['avg_batch_size']:.2f}, "
                        f"Timeout batches={batch_stats['timeout_batches']}, "
                        f"Queue={batch_stats['queue_size']}/{batch_stats['queue_capacity']}, "
                        f"Avg queue wait={batch_stats['avg_queue_wait_ms']:.2f}ms, "
                        f"Avg postprocess wait={batch_stats['avg_postprocess_wait_ms']:.2f}ms"
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
        if self.settings.yolo.batch_inference_enabled:
            self._inference_task = asyncio.create_task(self._inference_batch_loop())
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

            if self._inference_task and not self._inference_task.done():
                await self._inference_task

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

        if self._inference_task:
            try:
                await asyncio.wait_for(self.inference_queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Inference queue drain timeout")

            if not self._inference_task.done():
                await self._inference_task

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
