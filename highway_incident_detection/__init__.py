"""Highway Incident Detection System.

An industrial-grade system for detecting incidents on highways using YOLO-based
computer vision algorithms.
"""

from highway_incident_detection.detectors import (
    EventDetectionContext,
    EventDetectorHandler,
    EventTypeLike,
    RegisteredEventDetector,
    load_detector_handler,
    register_detectors_from_config,
)
from highway_incident_detection.output.builtin_handlers import (
    HTTPWebhookHandler,
    KafkaHandler,
    RedisHandler,
)
from highway_incident_detection.output.result_output import EventResult, ResultHandler
from highway_incident_detection.system import HighwayIncidentDetectionSystem

__version__ = "1.0.0"

__all__ = [
    "EventResult",
    "EventDetectionContext",
    "EventDetectorHandler",
    "EventTypeLike",
    "HTTPWebhookHandler",
    "HighwayIncidentDetectionSystem",
    "KafkaHandler",
    "RegisteredEventDetector",
    "RedisHandler",
    "ResultHandler",
    "load_detector_handler",
    "register_detectors_from_config",
    "__version__",
]
