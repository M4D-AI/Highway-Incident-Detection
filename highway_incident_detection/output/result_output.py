"""Standardized result output objects and handler types."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from highway_incident_detection.detectors.event_types import IncidentEvent


@dataclass
class EventResult:
    """Normalized event result emitted after persistence and audit."""

    event: IncidentEvent
    event_id: str
    keyframe_path: str
    video_path: str
    summary_path: str
    audit_path: str = ""
    audit: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event result for downstream systems."""
        return {
            "event_id": self.event_id,
            "event_type": self.event.event_type_value,
            "stream_name": self.event.stream_name,
            "timestamp": self.event.timestamp.isoformat(),
            "frame_number": self.event.frame_number,
            "confidence": self.event.confidence,
            "description": self.event.description,
            "bbox": list(self.event.bbox) if self.event.bbox else None,
            "metadata": self.event.metadata,
            "keyframe_path": self.keyframe_path,
            "video_path": self.video_path,
            "summary_path": self.summary_path,
            "audit_path": self.audit_path,
            "audit": self.audit,
        }


ResultHandler = Callable[[EventResult], Union[None, Awaitable[None]]]
