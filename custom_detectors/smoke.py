"""Example custom detector for smoke-like abnormal scenes."""

from highway_incident_detection.detectors import EventDetectionContext


def smoke_alert_detector(detector, ctx: EventDetectionContext):
    """Example custom detector loaded from configuration.

    This rule is intentionally simple: it emits one synthetic smoke alert every
    100 frames so users can understand the extension contract quickly.
    """
    if detector.is_in_cooldown(ctx.stream_name, "smoke_alert", ctx.timestamp):
        return None

    if ctx.frame_number % 100 != 0:
        return None

    detector.update_event_time(ctx.stream_name, "smoke_alert", ctx.timestamp)
    return detector.create_event(
        event_type="smoke_alert",
        stream_name=ctx.stream_name,
        timestamp=ctx.timestamp,
        frame_number=ctx.frame_number,
        confidence=0.9,
        description="Example smoke alert loaded from custom detector module",
        metadata={
            "source": "custom_detectors.smoke",
            "rule_type": "demo",
        },
    )
