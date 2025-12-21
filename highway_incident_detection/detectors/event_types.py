"""Event type definitions."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime


class EventType(Enum):
    """Event type enumeration."""

    CONSTRUCTION = "construction"
    REVERSE_VEHICLE = "reverse_vehicle"
    MOTORCYCLE = "motorcycle"
    PEDESTRIAN = "pedestrian"
    ROAD_DEBRIS = "road_debris"


@dataclass
class IncidentEvent:
    """Incident event data structure."""

    event_type: EventType
    stream_name: str
    timestamp: datetime
    frame_number: int
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None
    description: str = ""
    metadata: dict = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

