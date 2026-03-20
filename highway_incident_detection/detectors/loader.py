"""Helpers for loading custom event detectors from configuration."""

from importlib import import_module
from typing import List, Sequence

from highway_incident_detection.config.settings import CustomEventDetectorConfig
from highway_incident_detection.detectors.event_detector import EventDetectorHandler


def load_detector_handler(handler_path: str) -> EventDetectorHandler:
    """Load a detector handler from a module:callable path."""
    module_name, separator, attribute_path = handler_path.partition(":")
    if not module_name or not separator or not attribute_path:
        raise ValueError(
            f"Invalid handler path '{handler_path}'. Expected format: module:callable"
        )

    module = import_module(module_name)
    target = module
    for attribute_name in attribute_path.split("."):
        target = getattr(target, attribute_name)

    if not callable(target):
        raise TypeError(f"Loaded handler '{handler_path}' is not callable")

    return target


def register_detectors_from_config(
    event_detector,
    detector_configs: Sequence[CustomEventDetectorConfig],
) -> List[str]:
    """Register configured custom detectors onto an EventDetector."""
    registered_names = []
    for detector_config in detector_configs:
        if not detector_config.enabled:
            continue

        handler = load_detector_handler(detector_config.handler)
        event_detector.register_detector(
            name=detector_config.name,
            event_type=detector_config.event_type,
            handler=handler,
            prepend=detector_config.prepend,
        )
        registered_names.append(detector_config.name)
    return registered_names
