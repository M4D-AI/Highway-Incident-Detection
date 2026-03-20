from datetime import datetime

from highway_incident_detection.config.settings import EventConfig
from highway_incident_detection.detectors import EventDetectionContext, EventDetector


def test_custom_event_detector_can_be_registered_and_removed():
    detector = EventDetector(EventConfig())

    def custom_detector(detector_instance: EventDetector, ctx: EventDetectionContext):
        if detector_instance.is_in_cooldown(
            ctx.stream_name,
            "smoke_alert",
            ctx.timestamp,
        ):
            return None
        detector_instance.update_event_time(
            ctx.stream_name,
            "smoke_alert",
            ctx.timestamp,
        )
        return detector_instance.create_event(
            event_type="smoke_alert",
            stream_name=ctx.stream_name,
            timestamp=ctx.timestamp,
            frame_number=ctx.frame_number,
            confidence=0.92,
            description="Custom smoke alert",
            metadata={"source": "custom"},
        )

    detector.register_detector(
        name="smoke_alert_rule",
        event_type="smoke_alert",
        handler=custom_detector,
        prepend=True,
    )

    events = detector.process_frame(
        detections=[],
        tracked_objects={},
        stream_name="cam_1",
        frame_number=1,
        timestamp=datetime.now(),
    )

    assert events[0].event_type_value == "smoke_alert"
    assert events[0].metadata["source"] == "custom"
    assert any(item.name == "smoke_alert_rule" for item in detector.list_detectors())

    detector.unregister_detector("smoke_alert_rule")
    assert all(item.name != "smoke_alert_rule" for item in detector.list_detectors())
