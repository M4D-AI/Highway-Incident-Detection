import asyncio
from datetime import datetime

from highway_incident_detection.audit import EventAuditor
from highway_incident_detection.config.settings import AuditConfig
from highway_incident_detection.detectors.event_types import EventType, IncidentEvent


def test_auditor_falls_back_to_local_policy_without_remote_endpoint():
    auditor = EventAuditor(
        AuditConfig(
            enabled=True,
            endpoint=None,
            api_key=None,
            min_event_confidence=0.35,
        )
    )
    event = IncidentEvent(
        event_type=EventType.PEDESTRIAN,
        stream_name="cam_1",
        timestamp=datetime.now(),
        frame_number=42,
        confidence=0.91,
        description="Pedestrian detected near emergency lane",
    )

    result = asyncio.run(auditor.audit_event(event, keyframe_path="", video_path=""))

    assert result.confirmed is True
    assert result.risk_level == "high"
    assert result.reviewer == "fallback_policy"
    assert "local risk policy" in result.reason


def test_auditor_supports_custom_prompt_template():
    auditor = EventAuditor(
        AuditConfig(
            enabled=True,
            endpoint="https://example.com/v1/chat/completions",
            api_key="test-key",
            system_prompt="You are a strict traffic reviewer.",
            user_prompt_template=(
                "事件={event_type}; 置信度={confidence:.2f}; "
                "流={stream_name}; 元数据={metadata_json}"
            ),
        )
    )
    event = IncidentEvent(
        event_type=EventType.ROAD_DEBRIS,
        stream_name="cam_2",
        timestamp=datetime.now(),
        frame_number=18,
        confidence=0.76,
        description="Debris detected on lane",
        metadata={"lane": "L2"},
    )

    payload = auditor._build_openai_payload(event, keyframe_path="", video_path="")

    assert payload["messages"][0]["content"] == "You are a strict traffic reviewer."
    assert payload["messages"][1]["content"][0]["text"].startswith("事件=road_debris")
    assert '"lane": "L2"' in payload["messages"][1]["content"][0]["text"]


def test_auditor_supports_prompt_files(tmp_path):
    system_prompt_path = tmp_path / "audit_system_prompt.txt"
    user_prompt_path = tmp_path / "audit_user_prompt.txt"
    system_prompt_path.write_text("System prompt from file", encoding="utf-8")
    user_prompt_path.write_text(
        "File prompt: {event_type} at frame {frame_number}",
        encoding="utf-8",
    )

    auditor = EventAuditor(
        AuditConfig(
            enabled=True,
            endpoint="https://example.com/v1/chat/completions",
            api_key="test-key",
            system_prompt_path=str(system_prompt_path),
            user_prompt_path=str(user_prompt_path),
        )
    )
    event = IncidentEvent(
        event_type=EventType.PEDESTRIAN,
        stream_name="cam_1",
        timestamp=datetime.now(),
        frame_number=7,
        confidence=0.88,
        description="Pedestrian detected near lane edge",
    )

    payload = auditor._build_openai_payload(event, keyframe_path="", video_path="")

    assert payload["messages"][0]["content"] == "System prompt from file"
    assert payload["messages"][1]["content"][0]["text"] == "File prompt: pedestrian at frame 7"


def test_auditor_supports_per_event_prompt_override():
    auditor = EventAuditor(
        AuditConfig(
            enabled=True,
            endpoint="https://example.com/v1/chat/completions",
            api_key="test-key",
            system_prompt="default system",
            user_prompt_template="default prompt for {event_type}",
            event_prompt_overrides={
                "pedestrian": {
                    "system_prompt": "pedestrian system",
                    "user_prompt_template": "pedestrian prompt {frame_number}",
                }
            },
        )
    )
    pedestrian_event = IncidentEvent(
        event_type=EventType.PEDESTRIAN,
        stream_name="cam_1",
        timestamp=datetime.now(),
        frame_number=21,
        confidence=0.92,
        description="Pedestrian detected",
    )
    debris_event = IncidentEvent(
        event_type=EventType.ROAD_DEBRIS,
        stream_name="cam_1",
        timestamp=datetime.now(),
        frame_number=22,
        confidence=0.85,
        description="Debris detected",
    )

    pedestrian_payload = auditor._build_openai_payload(
        pedestrian_event, keyframe_path="", video_path=""
    )
    debris_payload = auditor._build_openai_payload(
        debris_event, keyframe_path="", video_path=""
    )

    assert pedestrian_payload["messages"][0]["content"] == "pedestrian system"
    assert pedestrian_payload["messages"][1]["content"][0]["text"] == "pedestrian prompt 21"
    assert debris_payload["messages"][0]["content"] == "default system"
    assert debris_payload["messages"][1]["content"][0]["text"] == "default prompt for road_debris"
