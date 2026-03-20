from datetime import datetime

from highway_incident_detection.config.settings import (
    CustomEventDetectorConfig,
    EventConfig,
)
from highway_incident_detection.detectors import (
    EventDetectionContext,
    EventDetector,
    load_detector_handler,
    register_detectors_from_config,
)


def test_custom_detector_loader_imports_and_registers(monkeypatch, tmp_path):
    module_dir = tmp_path / "custom_detectors"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "smoke.py").write_text(
        "\n".join(
            [
                "def smoke_alert_detector(detector, ctx):",
                "    if detector.is_in_cooldown(ctx.stream_name, 'smoke_alert', ctx.timestamp):",
                "        return None",
                "    detector.update_event_time(ctx.stream_name, 'smoke_alert', ctx.timestamp)",
                "    return detector.create_event(",
                "        event_type='smoke_alert',",
                "        stream_name=ctx.stream_name,",
                "        timestamp=ctx.timestamp,",
                "        frame_number=ctx.frame_number,",
                "        confidence=0.95,",
                "        description='Loaded from config',",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    handler = load_detector_handler("custom_detectors.smoke:smoke_alert_detector")
    detector = EventDetector(EventConfig())
    register_detectors_from_config(
        detector,
        [
            CustomEventDetectorConfig(
                name="smoke_alert_rule",
                event_type="smoke_alert",
                handler="custom_detectors.smoke:smoke_alert_detector",
                enabled=True,
                prepend=True,
            )
        ],
    )

    context = EventDetectionContext(
        detections=[],
        tracked_objects={},
        stream_name="cam_1",
        frame_number=3,
        timestamp=datetime.now(),
    )
    loaded_event = handler(detector, context)
    events = detector.process_frame(
        detections=[],
        tracked_objects={},
        stream_name="cam_2",
        frame_number=4,
        timestamp=datetime.now(),
    )

    assert loaded_event.event_type_value == "smoke_alert"
    assert events[0].event_type_value == "smoke_alert"
    assert detector.list_detectors()[0].name == "smoke_alert_rule"
