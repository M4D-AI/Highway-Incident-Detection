from highway_incident_detection import (
    EventDetectionContext,
    EventDetectorHandler,
    EventResult,
    HTTPWebhookHandler,
    HighwayIncidentDetectionSystem,
    KafkaHandler,
    RegisteredEventDetector,
    RedisHandler,
    load_detector_handler,
    register_detectors_from_config,
)


def test_system_is_exported_from_package():
    assert HighwayIncidentDetectionSystem.__name__ == "HighwayIncidentDetectionSystem"
    assert EventResult.__name__ == "EventResult"
    assert EventDetectionContext.__name__ == "EventDetectionContext"
    assert EventDetectorHandler.__name__ == "Callable"
    assert HTTPWebhookHandler.__name__ == "HTTPWebhookHandler"
    assert KafkaHandler.__name__ == "KafkaHandler"
    assert RegisteredEventDetector.__name__ == "RegisteredEventDetector"
    assert RedisHandler.__name__ == "RedisHandler"
    assert load_detector_handler.__name__ == "load_detector_handler"
    assert register_detectors_from_config.__name__ == "register_detectors_from_config"
