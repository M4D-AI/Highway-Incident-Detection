"""Event detection logic."""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
from highway_incident_detection.detectors.event_types import EventType, IncidentEvent
from highway_incident_detection.config.settings import EventConfig
from highway_incident_detection.utils.tracker import TrackedObject


class EventDetector:
    """Detect various highway incidents."""

    VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})

    def __init__(self, config: EventConfig):
        """Initialize event detector.

        Args:
            config: Event detection configuration.
        """
        self.config = config

        # Event state tracking
        self.last_event_time: Dict[Tuple[str, EventType], datetime] = {}
        self.construction_frame_count: Dict[str, int] = defaultdict(int)
        self.reverse_tracks: Dict[str, Dict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=20))
        )
        # Parking detection: track vehicle positions over time
        self.parking_tracks: Dict[str, Dict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=100))
        )
        # Congestion detection: track vehicle counts and speeds
        self.congestion_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

    def _prune_track_state(self, stream_name: str, active_track_ids: List[int]) -> None:
        """Remove stale per-track state for disappeared objects."""
        active_ids = set(active_track_ids)
        for state_map in (self.reverse_tracks[stream_name], self.parking_tracks[stream_name]):
            stale_ids = [track_id for track_id in state_map.keys() if track_id not in active_ids]
            for track_id in stale_ids:
                del state_map[track_id]

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
        self._prune_track_state(stream_name, list(tracked_objects.keys()))
        for track_id, obj in tracked_objects.items():
            # Only track vehicles (class IDs 2, 3, 5, 7)
            if obj.class_id not in self.VEHICLE_CLASS_IDS:
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

    def detect_abnormal_parking(
        self,
        tracked_objects: Dict[int, TrackedObject],
        stream_name: str,
        frame_number: int,
        timestamp: datetime,
    ) -> Optional[IncidentEvent]:
        """Detect abnormal parking events (vehicles stationary for too long).

        Args:
            tracked_objects: Dictionary of tracked objects.
            stream_name: Name of the stream.
            frame_number: Current frame number.
            timestamp: Current timestamp.

        Returns:
            IncidentEvent if abnormal parking detected, None otherwise.
        """
        if self._is_in_cooldown(stream_name, EventType.ABNORMAL_PARKING, timestamp):
            return None

        # Track vehicle positions
        self._prune_track_state(stream_name, list(tracked_objects.keys()))
        for track_id, obj in tracked_objects.items():
            # Only track vehicles (class IDs 2, 3, 5, 7)
            if obj.class_id not in self.VEHICLE_CLASS_IDS:
                continue

            # Calculate speed magnitude
            speed = np.sqrt(obj.velocity[0] ** 2 + obj.velocity[1] ** 2)

            # Store position and speed history
            self.parking_tracks[stream_name][track_id].append({
                "center": obj.center,
                "speed": speed,
                "frame": frame_number,
            })

            # Check if vehicle has been stationary for too long
            track_history = list(self.parking_tracks[stream_name][track_id])
            if len(track_history) >= self.config.parking_detection_frames:
                # Check if vehicle is stationary
                recent_positions = track_history[-self.config.parking_detection_frames:]
                speeds = [item["speed"] for item in recent_positions]

                # Check average speed
                avg_speed = np.mean(speeds)
                if avg_speed > self.config.parking_speed_threshold:
                    continue  # Vehicle is moving

                # Check position stability
                positions = [item["center"] for item in recent_positions]
                first_pos = positions[0]
                last_pos = positions[-1]
                position_change = np.sqrt(
                    (last_pos[0] - first_pos[0]) ** 2 + (last_pos[1] - first_pos[1]) ** 2
                )

                if position_change > self.config.parking_position_threshold:
                    continue  # Vehicle has moved significantly

                # Vehicle is stationary for too long - abnormal parking detected
                self._update_event_time(stream_name, EventType.ABNORMAL_PARKING, timestamp)
                return IncidentEvent(
                    event_type=EventType.ABNORMAL_PARKING,
                    stream_name=stream_name,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    confidence=0.85,  # Based on consistent stationary behavior
                    bbox=obj.bbox,
                    description=f"Abnormal parking detected (track_id: {track_id}, stationary for {len(recent_positions)} frames)",
                    metadata={
                        "track_id": track_id,
                        "average_speed": float(avg_speed),
                        "position_change": float(position_change),
                        "stationary_frames": len(recent_positions),
                    },
                )

        return None

    def detect_congestion(
        self,
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
        tracked_objects: Dict[int, TrackedObject],
        stream_name: str,
        frame_number: int,
        timestamp: datetime,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Optional[IncidentEvent]:
        """Detect traffic congestion events.

        Args:
            detections: List of (class_id, confidence, bbox) detections.
            tracked_objects: Dictionary of tracked objects.
            stream_name: Name of the stream.
            frame_number: Current frame number.
            timestamp: Current timestamp.
            frame_width: Frame width for density calculation (optional).
            frame_height: Frame height for density calculation (optional).

        Returns:
            IncidentEvent if congestion detected, None otherwise.
        """
        if self._is_in_cooldown(stream_name, EventType.CONGESTION, timestamp):
            return None

        # Filter vehicles only (class IDs 2, 3, 5, 7: car, motorcycle, bus, truck)
        vehicle_detections = [
            d for d in detections
            if d[0] in self.VEHICLE_CLASS_IDS
        ]

        if len(vehicle_detections) < self.config.congestion_min_vehicles:
            # Reset congestion history if not enough vehicles
            self.congestion_history[stream_name].clear()
            return None

        # Calculate average speed of tracked vehicles
        vehicle_speeds = []
        for track_id, obj in tracked_objects.items():
            if obj.class_id in self.VEHICLE_CLASS_IDS:  # Only vehicles
                speed = np.sqrt(obj.velocity[0] ** 2 + obj.velocity[1] ** 2)
                vehicle_speeds.append(speed)

        avg_speed = np.mean(vehicle_speeds) if vehicle_speeds else float('inf')

        # Calculate vehicle density if frame dimensions provided
        density = 0.0
        if frame_width and frame_height:
            frame_area = frame_width * frame_height
            if frame_area > 0:
                # Approximate vehicle area (sum of bbox areas)
                vehicle_area = sum(
                    (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    for _, _, bbox in vehicle_detections
                )
                density = vehicle_area / frame_area

        # Record congestion metrics
        congestion_metrics = {
            "vehicle_count": len(vehicle_detections),
            "avg_speed": float(avg_speed),
            "density": float(density),
            "frame": frame_number,
        }
        self.congestion_history[stream_name].append(congestion_metrics)

        # Check if congestion conditions are met
        history = list(self.congestion_history[stream_name])
        if len(history) >= self.config.congestion_detection_frames:
            recent_history = history[-self.config.congestion_detection_frames:]

            # Check if consistently congested
            congested_frames = 0
            total_vehicles = 0
            total_speed = 0.0
            total_density = 0.0

            for metrics in recent_history:
                total_vehicles += metrics["vehicle_count"]
                total_speed += metrics["avg_speed"]
                total_density += metrics["density"]

                # Check if this frame meets congestion criteria
                if (
                    metrics["vehicle_count"] >= self.config.congestion_min_vehicles
                    and metrics["avg_speed"] <= self.config.congestion_speed_threshold
                    and (
                        not frame_width or not frame_height
                        or metrics["density"] >= self.config.congestion_density_threshold
                    )
                ):
                    congested_frames += 1

            # Calculate averages
            avg_vehicle_count = total_vehicles / len(recent_history)
            avg_speed_over_time = total_speed / len(recent_history)
            avg_density_over_time = total_density / len(recent_history)

            # Congestion threshold: at least 80% of frames meet criteria
            congestion_ratio = congested_frames / len(recent_history)
            if congestion_ratio >= 0.8:
                # Calculate representative bbox (center of all vehicles)
                if vehicle_detections:
                    all_x = []
                    all_y = []
                    for _, _, bbox in vehicle_detections:
                        all_x.extend([bbox[0], bbox[2]])
                        all_y.extend([bbox[1], bbox[3]])

                    if all_x and all_y:
                        center_bbox = (
                            min(all_x),
                            min(all_y),
                            max(all_x),
                            max(all_y),
                        )
                    else:
                        center_bbox = vehicle_detections[0][2]
                else:
                    center_bbox = None

                self._update_event_time(stream_name, EventType.CONGESTION, timestamp)
                return IncidentEvent(
                    event_type=EventType.CONGESTION,
                    stream_name=stream_name,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    confidence=min(0.95, 0.7 + congestion_ratio * 0.25),
                    bbox=center_bbox,
                    description=(
                        f"Traffic congestion detected: {int(avg_vehicle_count)} vehicles, "
                        f"avg speed {avg_speed_over_time:.2f} px/frame"
                    ),
                    metadata={
                        "vehicle_count": int(avg_vehicle_count),
                        "avg_speed": float(avg_speed_over_time),
                        "density": float(avg_density_over_time),
                        "congestion_ratio": float(congestion_ratio),
                        "congested_frames": congested_frames,
                    },
                )

        return None

    def process_frame(
        self,
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
        tracked_objects: Dict[int, TrackedObject],
        stream_name: str,
        frame_number: int,
        timestamp: datetime,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
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

        parking_event = self.detect_abnormal_parking(
            tracked_objects, stream_name, frame_number, timestamp
        )
        if parking_event:
            events.append(parking_event)

        congestion_event = self.detect_congestion(
            detections,
            tracked_objects,
            stream_name,
            frame_number,
            timestamp,
            frame_width,
            frame_height,
        )
        if congestion_event:
            events.append(congestion_event)

        return events
