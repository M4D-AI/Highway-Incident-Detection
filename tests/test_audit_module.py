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
