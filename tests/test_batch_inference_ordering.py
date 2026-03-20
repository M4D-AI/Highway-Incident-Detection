import asyncio
import threading
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from highway_incident_detection.system import (
    FrameEnvelope,
    FrameInferenceResult,
    HighwayIncidentDetectionSystem,
)


def test_drain_ready_stream_results_preserves_per_stream_order():
    system = HighwayIncidentDetectionSystem.__new__(HighwayIncidentDetectionSystem)
    system._stream_locks = {"cam_1": threading.Lock()}
    system._pending_inference_results = {
        "cam_1": {
            2: FrameInferenceResult(
                envelope=FrameEnvelope(
                    "cam_1",
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    2,
                    datetime.now(),
                ),
                detections=[],
                batch_size=2,
                batch_timed_out=False,
                queue_wait_ms=0.0,
                inference_duration_ms=0.0,
                ready_at_monotonic=0.0,
            ),
            1: FrameInferenceResult(
                envelope=FrameEnvelope(
                    "cam_1",
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    1,
                    datetime.now(),
                ),
                detections=[],
                batch_size=2,
                batch_timed_out=False,
                queue_wait_ms=0.0,
                inference_duration_ms=0.0,
                ready_at_monotonic=0.0,
            ),
        }
    }
    system._next_frame_to_process = {"cam_1": 1}
    system._batch_metrics_lock = threading.Lock()
    system._batch_metrics = HighwayIncidentDetectionSystem._empty_batch_metrics_state()

    processed = []

    def finalize(stream_lock, envelope, detections):
        processed.append(envelope.frame_number)
        return []

    async def process_events(events):
        return None

    system._finalize_inference_result = finalize
    system._process_detected_events = process_events

    asyncio.run(system._drain_ready_stream_results("cam_1"))

    assert processed == [1, 2]
    assert system._next_frame_to_process["cam_1"] == 3


def test_process_frame_uses_batch_queue_when_enabled():
    system = HighwayIncidentDetectionSystem.__new__(HighwayIncidentDetectionSystem)
    system.settings = SimpleNamespace(
        yolo=SimpleNamespace(batch_inference_enabled=True),
    )

    captured = {}

    async def enqueue(stream_name, frame, frame_number, timestamp):
        captured["stream_name"] = stream_name
        captured["frame_number"] = frame_number
        captured["timestamp"] = timestamp

    system._enqueue_frame_for_batch_inference = enqueue

    asyncio.run(
        system._process_frame(
            "cam_1",
            np.zeros((4, 4, 3), dtype=np.uint8),
            7,
            datetime.now(),
        )
    )

    assert captured["stream_name"] == "cam_1"
    assert captured["frame_number"] == 7


def test_run_inference_batch_fills_missing_results():
    system = HighwayIncidentDetectionSystem.__new__(HighwayIncidentDetectionSystem)
    system._inference_lock = threading.Lock()
    system.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    system.yolo_model = SimpleNamespace(
        predict_batch=lambda images, verbose=False: [[]],
    )

    batch = [
        FrameEnvelope("cam_1", np.zeros((4, 4, 3), dtype=np.uint8), 1, datetime.now()),
        FrameEnvelope("cam_2", np.zeros((4, 4, 3), dtype=np.uint8), 2, datetime.now()),
    ]

    results = asyncio.run(system._run_inference_batch(batch))

    assert len(results) == 2
    assert results[0] == []
    assert results[1] == []


def test_inference_batch_loop_does_not_double_ack_on_finalize_error():
    system = HighwayIncidentDetectionSystem.__new__(HighwayIncidentDetectionSystem)
    system.running = False
    system.settings = SimpleNamespace(
        yolo=SimpleNamespace(
            batch_timeout_ms=1,
            max_batch_size=4,
        )
    )
    system.inference_queue = asyncio.Queue()
    system.inference_queue.put_nowait(
        FrameEnvelope("cam_1", np.zeros((4, 4, 3), dtype=np.uint8), 1, datetime.now())
    )
    system._pending_inference_results = {}
    system._record_batch_completion = lambda *args, **kwargs: None
    system._run_inference_batch = lambda batch: asyncio.sleep(0, result=[[]])
    system._drain_ready_stream_results = lambda stream_name: asyncio.sleep(
        0, result=(_ for _ in ()).throw(RuntimeError("boom"))
    )
    errors = []
    system.logger = SimpleNamespace(
        error=lambda message: errors.append(message),
        warning=lambda *args, **kwargs: None,
    )

    asyncio.run(system._inference_batch_loop())
    asyncio.run(asyncio.wait_for(system.inference_queue.join(), timeout=0.1))

    assert errors
