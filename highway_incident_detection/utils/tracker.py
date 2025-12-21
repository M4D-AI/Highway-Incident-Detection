"""Object tracking utility for motion analysis."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from highway_incident_detection.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedObject:
    """Tracked object information."""

    track_id: int
    class_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]
    velocity: Tuple[float, float]  # pixels per frame
    frame_count: int
    last_seen: int


class ObjectTracker:
    """Simple object tracker using center point matching."""

    def __init__(
        self,
        max_disappeared: int = 5,
        max_distance: float = 50.0,
    ):
        """Initialize object tracker.

        Args:
            max_disappeared: Maximum frames an object can be missing.
            max_distance: Maximum distance for matching objects.
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        self.objects: Dict[int, TrackedObject] = {}
        self.disappeared: Dict[int, int] = {}

    def _calculate_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculate center point of bounding box.

        Args:
            bbox: Bounding box (x1, y1, x2, y2).

        Returns:
            Center point (cx, cy).
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _calculate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points.

        Args:
            point1: First point (x, y).
            point2: Second point (x, y).

        Returns:
            Euclidean distance.
        """
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def update(
        self,
        detections: List[Tuple[int, Tuple[float, float, float, float]]],
        frame_number: int,
    ) -> Dict[int, TrackedObject]:
        """Update tracker with new detections.

        Args:
            detections: List of (class_id, bbox) tuples.
            frame_number: Current frame number.

        Returns:
            Dictionary of tracked objects.
        """
        if len(detections) == 0:
            # Mark all objects as disappeared
            for track_id in list(self.objects.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.objects[track_id]
                    del self.disappeared[track_id]
            return self.objects

        # Calculate centers for all detections
        detection_centers = [
            (i, self._calculate_center(bbox), class_id, bbox)
            for i, (class_id, bbox) in enumerate(detections)
        ]

        # Match existing objects with detections
        if len(self.objects) == 0:
            # First frame: create new tracks
            for _, center, class_id, bbox in detection_centers:
                track_id = self.next_id
                self.next_id += 1
                self.objects[track_id] = TrackedObject(
                    track_id=track_id,
                    class_id=class_id,
                    bbox=bbox,
                    center=center,
                    velocity=(0.0, 0.0),
                    frame_count=1,
                    last_seen=frame_number,
                )
                self.disappeared[track_id] = 0
        else:
            # Match detections to existing tracks
            matched_indices = set()
            matched_tracks = set()

            # Calculate distances and match
            for track_id, obj in self.objects.items():
                best_match = None
                best_distance = float("inf")

                for det_idx, center, class_id, bbox in detection_centers:
                    if det_idx in matched_indices:
                        continue
                    if obj.class_id != class_id:
                        continue

                    distance = self._calculate_distance(obj.center, center)
                    if distance < best_distance and distance < self.max_distance:
                        best_distance = distance
                        best_match = (det_idx, center, class_id, bbox)

                if best_match:
                    det_idx, center, class_id, bbox = best_match
                    matched_indices.add(det_idx)
                    matched_tracks.add(track_id)

                    # Update object
                    velocity = (
                        center[0] - obj.center[0],
                        center[1] - obj.center[1],
                    )
                    self.objects[track_id] = TrackedObject(
                        track_id=track_id,
                        class_id=class_id,
                        bbox=bbox,
                        center=center,
                        velocity=velocity,
                        frame_count=obj.frame_count + 1,
                        last_seen=frame_number,
                    )
                    self.disappeared[track_id] = 0

            # Handle unmatched detections (new objects)
            for det_idx, center, class_id, bbox in detection_centers:
                if det_idx not in matched_indices:
                    track_id = self.next_id
                    self.next_id += 1
                    self.objects[track_id] = TrackedObject(
                        track_id=track_id,
                        class_id=class_id,
                        bbox=bbox,
                        center=center,
                        velocity=(0.0, 0.0),
                        frame_count=1,
                        last_seen=frame_number,
                    )
                    self.disappeared[track_id] = 0

            # Handle unmatched tracks (disappeared objects)
            for track_id in list(self.objects.keys()):
                if track_id not in matched_tracks:
                    self.disappeared[track_id] += 1
                    if self.disappeared[track_id] > self.max_disappeared:
                        del self.objects[track_id]
                        del self.disappeared[track_id]

        return self.objects

    def get_objects(self) -> Dict[int, TrackedObject]:
        """Get all currently tracked objects.

        Returns:
            Dictionary of tracked objects.
        """
        return self.objects.copy()

