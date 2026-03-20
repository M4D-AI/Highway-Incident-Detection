"""Event detection modules."""

from highway_incident_detection.detectors.event_detector import (
    EventDetectionContext,
    EventDetector,
    EventDetectorHandler,
    RegisteredEventDetector,
)
from highway_incident_detection.detectors.loader import (
    load_detector_handler,
    register_detectors_from_config,
)
from highway_incident_detection.detectors.event_types import EventType, EventTypeLike, event_type_name

__all__ = [
    "EventDetectionContext",
    "EventDetector",
    "EventDetectorHandler",
    "EventType",
    "EventTypeLike",
    "RegisteredEventDetector",
    "event_type_name",
    "load_detector_handler",
    "register_detectors_from_config",
]
