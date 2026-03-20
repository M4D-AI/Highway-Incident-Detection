import asyncio
import json
import threading
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from highway_incident_detection.config.settings import OutputConfig
from highway_incident_detection.detectors.event_types import EventType, IncidentEvent
from highway_incident_detection.summary.summary_generator import SummaryGenerator
from highway_incident_detection.system import (
    FrameEnvelope,
    FrameInferenceResult,
    HighwayIncidentDetectionSystem,
)


def test_get_batch_inference_stats_aggregates_metrics():
    system = HighwayIncidentDetectionSystem.__new__(HighwayIncidentDetectionSystem)
    system.settings = SimpleNamespace(
        yolo=SimpleNamespace(
            batch_inference_enabled=True,
            batch_queue_size=64,
            max_batch_size=8,
        )
    )
    system.inference_queue = asyncio.Queue()
    system._batch_metrics_lock = threading.Lock()
    system._batch_metrics = {
        "total_enqueued_frames": 6,
        "total_completed_frames": 4,
        "total_batches": 2,
        "total_frames_in_batches": 4,
        "timeout_batches": 1,
        "full_batches": 1,
        "max_batch_size_seen": 3,
        "max_queue_depth_seen": 5,
        "total_queue_wait_ms": 18.0,
        "total_inference_duration_ms": 24.0,
        "total_postprocess_wait_ms": 10.0,
        "per_stream": {
            "cam_1": {
                "enqueued_frames": 3,
                "completed_frames": 2,
                "last_frame_number": 12,
                "last_queue_wait_ms": 4.0,
                "last_postprocess_wait_ms": 2.0,
                "total_queue_wait_ms": 9.0,
                "total_postprocess_wait_ms": 5.0,
            }
        },
    }

    stats = system.get_batch_inference_stats()

    assert stats["avg_batch_size"] == 2.0
    assert stats["timeout_rate"] == 0.5
    assert stats["avg_queue_wait_ms"] == 4.5
    assert stats["avg_inference_duration_ms"] == 12.0
    assert stats["per_stream"]["cam_1"]["avg_postprocess_wait_ms"] == 2.5


def test_drain_ready_stream_results_attaches_processing_metrics():
    system = HighwayIncidentDetectionSystem.__new__(HighwayIncidentDetectionSystem)
    system._stream_locks = {"cam_1": threading.Lock()}
    system._pending_inference_results = {
        "cam_1": {
            1: FrameInferenceResult(
                envelope=FrameEnvelope(
                    "cam_1",
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    1,
                    datetime.now(),
                    enqueued_at_monotonic=1.0,
                    queue_depth_at_enqueue=2,
                ),
                detections=[],
                batch_size=3,
                batch_timed_out=True,
                queue_wait_ms=6.5,
                inference_duration_ms=8.0,
                ready_at_monotonic=0.0,
            )
        }
    }
    system._next_frame_to_process = {"cam_1": 1}
    system._batch_metrics_lock = threading.Lock()
    system._batch_metrics = HighwayIncidentDetectionSystem._empty_batch_metrics_state()

    processed = []

    def finalize(stream_lock, envelope, detections):
        return [
            IncidentEvent(
                event_type=EventType.CONGESTION,
                stream_name=envelope.stream_name,
                timestamp=envelope.timestamp,
                frame_number=envelope.frame_number,
                confidence=0.9,
                description="Traffic congestion detected",
            )
        ]

    async def process_events(events):
        processed.extend(events)

    system._finalize_inference_result = finalize
    system._process_detected_events = process_events

    asyncio.run(system._drain_ready_stream_results("cam_1"))

    metrics = processed[0].metadata["processing_metrics"]
    assert metrics["mode"] == "micro_batch"
    assert metrics["batch_size"] == 3
    assert metrics["batch_timed_out"] is True
    assert metrics["queue_depth_at_enqueue"] == 2


def test_generate_summary_includes_processing_metrics(tmp_path):
    config = OutputConfig(
        output_dir=str(tmp_path),
        keyframe_dir=str(tmp_path / "keyframes"),
        video_dir=str(tmp_path / "videos"),
        summary_dir=str(tmp_path / "summaries"),
        audit_dir=str(tmp_path / "audits"),
    )
    generator = SummaryGenerator(config)
    event = IncidentEvent(
        event_type=EventType.CONGESTION,
        stream_name="cam_1",
        timestamp=datetime.now(),
        frame_number=11,
        confidence=0.9,
        description="Traffic congestion detected",
        metadata={"processing_metrics": {"mode": "micro_batch", "batch_size": 4}},
    )

    summary_path = asyncio.run(
        generator.generate_summary(
            event,
            "keyframe.jpg",
            "video.mp4",
            processing_metrics=event.metadata["processing_metrics"],
        )
    )

    summary = json.loads(open(summary_path, "r", encoding="utf-8").read())
    assert summary["processing_metrics"]["mode"] == "micro_batch"
    assert summary["processing_metrics"]["batch_size"] == 4
