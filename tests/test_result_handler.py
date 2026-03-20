import asyncio
from datetime import datetime

from highway_incident_detection.detectors.event_types import EventType, IncidentEvent
from highway_incident_detection.output.result_output import EventResult
from highway_incident_detection.summary.summary_generator import SummaryGenerator
from highway_incident_detection.config.settings import OutputConfig


def test_result_handler_is_called_after_result_emission(tmp_path):
    received = []

    def handler(result: EventResult) -> None:
        received.append(result.to_dict())

    config = OutputConfig(
        output_dir=str(tmp_path),
        keyframe_dir=str(tmp_path / "keyframes"),
        video_dir=str(tmp_path / "videos"),
        summary_dir=str(tmp_path / "summaries"),
        audit_dir=str(tmp_path / "audits"),
    )
    generator = SummaryGenerator(config, result_handlers=[handler])
    event = IncidentEvent(
        event_type=EventType.CONGESTION,
        stream_name="cam_1",
        timestamp=datetime.now(),
        frame_number=10,
        confidence=0.88,
        description="Traffic congestion detected",
    )
    result = EventResult(
        event=event,
        event_id="evt-1",
        keyframe_path="a.jpg",
        video_path="a.mp4",
        summary_path="a.json",
        audit_path="a.audit.json",
        audit={"confirmed": True, "risk_level": "medium"},
    )

    asyncio.run(generator._emit_result(result))

    assert len(received) == 1
    assert received[0]["event_id"] == "evt-1"
    assert received[0]["summary_path"] == "a.json"


def test_result_handlers_support_sync_and_async(tmp_path):
    received = []

    def sync_handler(result: EventResult) -> None:
        received.append(("sync", result.event_id))

    async def async_handler(result: EventResult) -> None:
        await asyncio.sleep(0.01)
        received.append(("async", result.event_id))

    config = OutputConfig(
        output_dir=str(tmp_path),
        keyframe_dir=str(tmp_path / "keyframes"),
        video_dir=str(tmp_path / "videos"),
        summary_dir=str(tmp_path / "summaries"),
        audit_dir=str(tmp_path / "audits"),
    )
    generator = SummaryGenerator(config, result_handlers=[sync_handler, async_handler])
    event = IncidentEvent(
        event_type=EventType.CONGESTION,
        stream_name="cam_1",
        timestamp=datetime.now(),
        frame_number=11,
        confidence=0.9,
        description="Traffic congestion detected",
    )
    result = EventResult(
        event=event,
        event_id="evt-2",
        keyframe_path="b.jpg",
        video_path="b.mp4",
        summary_path="b.json",
    )

    asyncio.run(generator._emit_result(result))

    assert ("sync", "evt-2") in received
    assert ("async", "evt-2") in received
