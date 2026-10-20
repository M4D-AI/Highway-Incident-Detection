import asyncio
import threading
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from highway_incident_detection.config.settings import Settings, StreamConfig
from highway_incident_detection.stream.rtsp_stream import RTSPStreamProcessor
from highway_incident_detection.system import (
    FrameEnvelope,
    FrameInferenceResult,
    HighwayIncidentDetectionSystem,
)


def _result(stream_name, frame_number):
    return FrameInferenceResult(
        envelope=FrameEnvelope(
            stream_name,
            np.zeros((4, 4, 3), dtype=np.uint8),
            frame_number,
            datetime.now(),
        ),
        detections=[],
        batch_size=1,
        batch_timed_out=False,
        queue_wait_ms=0.0,
        inference_duration_ms=0.0,
        ready_at_monotonic=0.0,
    )


def _drain_system(pending, next_frame):
    system = HighwayIncidentDetectionSystem.__new__(HighwayIncidentDetectionSystem)
    system._stream_locks = {"cam_1": threading.Lock()}
    system._pending_inference_results = {"cam_1": pending}
    system._next_frame_to_process = {"cam_1": next_frame}
    system._batch_metrics_lock = threading.Lock()
    system._batch_metrics = HighwayIncidentDetectionSystem._empty_batch_metrics_state()
    system.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

    processed = []

    def finalize(stream_lock, envelope, detections):
        processed.append(envelope.frame_number)
        return []

    async def process_events(events):
        return None

    system._finalize_inference_result = finalize
    system._process_detected_events = process_events
    return system, processed


def test_drain_skips_missing_frame_instead_of_stalling():
    # Frame 2 never arrives (e.g. its batch failed); 3 and 4 must still be processed.
    system, processed = _drain_system({3: _result("cam_1", 3), 4: _result("cam_1", 4)}, 2)

    asyncio.run(system._drain_ready_stream_results("cam_1"))

    assert processed == [3, 4]
    assert system._next_frame_to_process["cam_1"] == 5
    assert system._pending_inference_results["cam_1"] == {}


def test_drain_discards_results_older_than_next_frame():
    system, processed = _drain_system({1: _result("cam_1", 1), 5: _result("cam_1", 5)}, 3)

    asyncio.run(system._drain_ready_stream_results("cam_1"))

    assert processed == [5]
    assert system._next_frame_to_process["cam_1"] == 6


def test_run_inference_batch_does_not_block_event_loop():
    system = HighwayIncidentDetectionSystem.__new__(HighwayIncidentDetectionSystem)
    system._inference_lock = threading.Lock()
    system.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    loop_thread = threading.get_ident()
    predict_threads = []

    def predict(image, verbose=False):
        predict_threads.append(threading.get_ident())
        return []

    system.yolo_model = SimpleNamespace(predict=predict)
    batch = [FrameEnvelope("cam_1", np.zeros((4, 4, 3), dtype=np.uint8), 1, datetime.now())]

    results = asyncio.run(system._run_inference_batch(batch))

    assert results == [[]]
    assert predict_threads and predict_threads[0] != loop_thread


def test_dropped_frames_do_not_consume_frame_numbers():
    received = []

    async def callback(stream_name, frame, frame_number, timestamp):
        received.append(frame_number)

    processor = RTSPStreamProcessor(
        StreamConfig(
            streams=[{"name": "cam_1", "url": "fake", "enabled": True}],
            max_pending_frames=1,
            drop_frames_when_busy=True,
        ),
        frame_callback=callback,
    )
    accept = iter([True, False, False, True])

    async def wait_for_slot(stream_name):
        return next(accept)

    frames = iter([True, True, True, True])

    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            if next(frames, False):
                return True, np.zeros((4, 4, 3), dtype=np.uint8)
            processor.running = False
            return False, None

        def release(self):
            pass

    processor._wait_for_callback_slot = wait_for_slot
    processor.cap_dict["cam_1"] = FakeCapture()
    processor.running = True

    async def run():
        await processor._process_stream("cam_1", "fake")
        await asyncio.sleep(0)

    asyncio.run(run())

    assert received == [1, 2]


def _lane_system(lane):
    system = HighwayIncidentDetectionSystem.__new__(HighwayIncidentDetectionSystem)
    system._scaled_zone_cache = {}
    system.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    return system, HighwayIncidentDetectionSystem._normalize_lane_config(lane)


def test_roi_points_scale_to_frame_size():
    system, lane_config = _lane_system(
        {
            "enabled": True,
            "roi_reference_size": [640, 340],
            "zones": [{"name": "z", "roi_points": [[0, 0], [640, 0], [320, 340]]}],
        }
    )

    zones = system._get_scaled_zones("cam_1", lane_config, 1920, 1080)

    assert zones[0]["roi_points"] == [(0, 0), (1920, 0), (960, 1080)]
    # Cached for the same frame size
    assert system._get_scaled_zones("cam_1", lane_config, 1920, 1080) is zones


def test_roi_points_unchanged_without_reference_size():
    system, lane_config = _lane_system(
        {"enabled": True, "zones": [{"name": "z", "roi_points": [[10, 20], [30, 40], [50, 60]]}]}
    )

    zones = system._get_scaled_zones("cam_1", lane_config, 1920, 1080)

    assert zones[0]["roi_points"] == [(10, 20), (30, 40), (50, 60)]


def test_settings_accept_roi_reference_size(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "lane:\n  enabled: true\n  roi_reference_size: [1920, 1080]\n",
        encoding="utf-8",
    )

    settings = Settings.from_yaml(str(config_path))

    assert settings.lane.roi_reference_size == (1920, 1080)
