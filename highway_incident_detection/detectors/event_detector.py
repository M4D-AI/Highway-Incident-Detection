"""Event detection logic."""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
from highway_incident_detection.detectors.event_types import EventType, IncidentEvent
from highway_incident_detection.config.settings import EventConfig
from highway_incident_detection.utils.tracker import ObjectTracker, TrackedObject
from highway_incident_detection.utils.logger import get_logger

logger = get_logger(__name__)


class EventDetector:
    """Detect various highway incidents."""

    def __init__(self, config: EventConfig):
        """Initialize event detector.

        Args:
            config: Event detection configuration.
        """
        self.config = config
        self.tracker = ObjectTracker(max_disappeared=5, max_distance=50.0)

        # Event state tracking
        self.event_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_event_time: Dict[Tuple[str, EventType], datetime] = {}
        self.construction_frame_count: Dict[str, int] = defaultdict(int)
        self.reverse_tracks: Dict[str, Dict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=20))
        )

    def _is_in_cooldown(
        self, stream_name: str, event_type: EventType, current_time: datetime
    ) -> bool:
        """Check if event is in cooldown period.

        Args:
            stream_name: Name of the stream.
            event_type: Type of event.
            current_time: Current timestamp.

        Returns:
            True if in cooldown, False otherwise.
        """
        key = (stream_name, event_type)
        if key not in self.last_event_time:
            return False

        time_diff = current_time - self.last_event_time[key]
        return time_diff.total_seconds() < self.config.event_cooldown_seconds

    def _update_event_time(
        self, stream_name: str, event_type: EventType, current_time: datetime
    ) -> None:
        """Update last event time.

        Args:
            stream_name: Name of the stream.
            event_type: Type of event.
            current_time: Current timestamp.
        """
        key = (stream_name, event_type)
        self.last_event_time[key] = current_time

    def detect_construction(
        self,
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
        stream_name: str,
        frame_number: int,
        timestamp: datetime,
    ) -> Optional[IncidentEvent]:
        """Detect construction events (construction vehicles + workers).

        Args:
            detections: List of (class_id, confidence, bbox) detections.
            stream_name: Name of the stream.
            frame_number: Current frame number.
            timestamp: Current timestamp.

        Returns:
            IncidentEvent if construction detected, None otherwise.
        """
        if self._is_in_cooldown(stream_name, EventType.CONSTRUCTION, timestamp):
            return None

        construction_vehicles = [
            d for d in detections
            if d[0] in self.config.construction_vehicle_class_ids
        ]
        workers = [
            d for d in detections
            if d[0] == self.config.worker_class_id
        ]

        if len(construction_vehicles) > 0 and len(workers) > 0:
            self.construction_frame_count[stream_name] += 1
            if self.construction_frame_count[stream_name] >= self.config.construction_detection_frames:
                # Reset counter
                self.construction_frame_count[stream_name] = 0

                # Calculate average confidence
                avg_conf = (
                    sum(v[1] for v in construction_vehicles) +
                    sum(w[1] for w in workers)
                ) / (len(construction_vehicles) + len(workers))

                # Use first vehicle bbox as reference
                bbox = construction_vehicles[0][2]

                self._update_event_time(stream_name, EventType.CONSTRUCTION, timestamp)
                return IncidentEvent(
                    event_type=EventType.CONSTRUCTION,
                    stream_name=stream_name,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    confidence=avg_conf,
                    bbox=bbox,
                    description=f"Construction detected: {len(construction_vehicles)} vehicles, {len(workers)} workers",
                    metadata={
                        "vehicle_count": len(construction_vehicles),
                        "worker_count": len(workers),
                    },
                )
        else:
            # Reset counter if conditions not met
            self.construction_frame_count[stream_name] = 0

        return None

    def detect_reverse_vehicle(
        self,
        tracked_objects: Dict[int, TrackedObject],
        stream_name: str,
        frame_number: int,
        timestamp: datetime,
    ) -> Optional[IncidentEvent]:
        """Detect reverse vehicle events.

        Args:
            tracked_objects: Dictionary of tracked objects.
            stream_name: Name of the stream.
            frame_number: Current frame number.
            timestamp: Current timestamp.

        Returns:
            IncidentEvent if reverse vehicle detected, None otherwise.
        """
        if self._is_in_cooldown(stream_name, EventType.REVERSE_VEHICLE, timestamp):
            return None

        # Track vehicle velocities
        for track_id, obj in tracked_objects.items():
            # Only track vehicles (class IDs 2, 3, 5, 7)
            if obj.class_id not in [2, 3, 5, 7]:
                continue

            # Calculate speed (pixels per frame) in x direction
            speed_x = obj.velocity[0]

            # Store velocity history
            self.reverse_tracks[stream_name][track_id].append(speed_x)

            # Check if consistently moving backwards
            if len(self.reverse_tracks[stream_name][track_id]) >= self.config.reverse_detection_frames:
                velocities = list(self.reverse_tracks[stream_name][track_id])
                avg_velocity = np.mean(velocities)

                if avg_velocity < self.config.reverse_speed_threshold:
                    # Vehicle is reversing
                    self._update_event_time(stream_name, EventType.REVERSE_VEHICLE, timestamp)
                    return IncidentEvent(
                        event_type=EventType.REVERSE_VEHICLE,
                        stream_name=stream_name,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        confidence=0.8,  # Based on consistent reverse motion
                        bbox=obj.bbox,
                        description=f"Vehicle reversing detected (track_id: {track_id})",
                        metadata={
                            "track_id": track_id,
                            "average_velocity": float(avg_velocity),
                        },
                    )

        return None

    def detect_motorcycle(
        self,
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
        stream_name: str,
        frame_number: int,
        timestamp: datetime,
    ) -> Optional[IncidentEvent]:
        """Detect motorcycle events.

        Args:
            detections: List of (class_id, confidence, bbox) detections.
            stream_name: Name of the stream.
            frame_number: Current frame number.
            timestamp: Current timestamp.

        Returns:
            IncidentEvent if motorcycle detected, None otherwise.
        """
        if self._is_in_cooldown(stream_name, EventType.MOTORCYCLE, timestamp):
            return None

        motorcycles = [
            d for d in detections
            if d[0] == self.config.motorcycle_class_id
        ]

        if len(motorcycles) > 0:
            # Use highest confidence detection
            best_detection = max(motorcycles, key=lambda x: x[1])
            self._update_event_time(stream_name, EventType.MOTORCYCLE, timestamp)
            return IncidentEvent(
                event_type=EventType.MOTORCYCLE,
                stream_name=stream_name,
                timestamp=timestamp,
                frame_number=frame_number,
                confidence=best_detection[1],
                bbox=best_detection[2],
                description=f"Motorcycle detected (confidence: {best_detection[1]:.2f})",
                metadata={"count": len(motorcycles)},
            )

        return None

    def detect_pedestrian(
        self,
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
        stream_name: str,
        frame_number: int,
        timestamp: datetime,
    ) -> Optional[IncidentEvent]:
        """Detect pedestrian events.

        Args:
            detections: List of (class_id, confidence, bbox) detections.
            stream_name: Name of the stream.
            frame_number: Current frame number.
            timestamp: Current timestamp.

        Returns:
            IncidentEvent if pedestrian detected, None otherwise.
        """
        if self._is_in_cooldown(stream_name, EventType.PEDESTRIAN, timestamp):
            return None

        pedestrians = [
            d for d in detections
            if d[0] == self.config.pedestrian_class_id
        ]

        if len(pedestrians) > 0:
            # Use highest confidence detection
            best_detection = max(pedestrians, key=lambda x: x[1])
            self._update_event_time(stream_name, EventType.PEDESTRIAN, timestamp)
            return IncidentEvent(
                event_type=EventType.PEDESTRIAN,
                stream_name=stream_name,
                timestamp=timestamp,
                frame_number=frame_number,
                confidence=best_detection[1],
                bbox=best_detection[2],
                description=f"Pedestrian detected (confidence: {best_detection[1]:.2f})",
                metadata={"count": len(pedestrians)},
            )

        return None

    def detect_road_debris(
        self,
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
        stream_name: str,
        frame_number: int,
        timestamp: datetime,
    ) -> Optional[IncidentEvent]:
        """Detect road debris events.

        Args:
            detections: List of (class_id, confidence, bbox) detections.
            stream_name: Name of the stream.
            frame_number: Current frame number.
            timestamp: Current timestamp.

        Returns:
            IncidentEvent if road debris detected, None otherwise.
        """
        if self._is_in_cooldown(stream_name, EventType.ROAD_DEBRIS, timestamp):
            return None

        debris = [
            d for d in detections
            if d[0] in self.config.debris_class_ids
        ]

        if len(debris) > 0:
            # Use highest confidence detection
            best_detection = max(debris, key=lambda x: x[1])
            self._update_event_time(stream_name, EventType.ROAD_DEBRIS, timestamp)
            return IncidentEvent(
                event_type=EventType.ROAD_DEBRIS,
                stream_name=stream_name,
                timestamp=timestamp,
                frame_number=frame_number,
                confidence=best_detection[1],
                bbox=best_detection[2],
                description=f"Road debris detected (confidence: {best_detection[1]:.2f})",
                metadata={"count": len(debris)},
            )

        return None

    def process_frame(
        self,
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
        tracked_objects: Dict[int, TrackedObject],
        stream_name: str,
        frame_number: int,
        timestamp: datetime,
    ) -> List[IncidentEvent]:
        """Process frame and detect all events.

        Args:
            detections: List of (class_id, confidence, bbox) detections.
            tracked_objects: Dictionary of tracked objects.
            stream_name: Name of the stream.
            frame_number: Current frame number.
            timestamp: Current timestamp.

        Returns:
            List of detected events.
        """
        events = []

        # Detect all event types
        construction_event = self.detect_construction(
            detections, stream_name, frame_number, timestamp
        )
        if construction_event:
            events.append(construction_event)

        reverse_event = self.detect_reverse_vehicle(
            tracked_objects, stream_name, frame_number, timestamp
        )
        if reverse_event:
            events.append(reverse_event)

        motorcycle_event = self.detect_motorcycle(
            detections, stream_name, frame_number, timestamp
        )
        if motorcycle_event:
            events.append(motorcycle_event)

        pedestrian_event = self.detect_pedestrian(
            detections, stream_name, frame_number, timestamp
        )
        if pedestrian_event:
            events.append(pedestrian_event)

        debris_event = self.detect_road_debris(
            detections, stream_name, frame_number, timestamp
        )
        if debris_event:
            events.append(debris_event)

        return events

