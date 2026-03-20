"""Event type definitions."""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from datetime import datetime


class EventType(Enum):
    """Event type enumeration."""

    CONSTRUCTION = "construction"
    REVERSE_VEHICLE = "reverse_vehicle"
    MOTORCYCLE = "motorcycle"
    PEDESTRIAN = "pedestrian"
    ROAD_DEBRIS = "road_debris"
    ABNORMAL_PARKING = "abnormal_parking"
    CONGESTION = "congestion"


EventTypeLike = Union[EventType, str]


def event_type_name(event_type: EventTypeLike) -> str:
    """Normalize event type values to a string name."""
    if isinstance(event_type, EventType):
        return event_type.value
    return str(event_type)


@dataclass
class IncidentEvent:
    """Incident event data structure."""

    event_type: EventTypeLike
    stream_name: str
    timestamp: datetime
    frame_number: int
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None
    description: str = ""
    metadata: Optional[Dict[str, object]] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def event_type_value(self) -> str:
        """Return the normalized string value for the event type."""
        return event_type_name(self.event_type)
