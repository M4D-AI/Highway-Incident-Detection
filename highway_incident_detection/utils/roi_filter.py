"""ROI (Region of Interest) filtering utilities."""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from highway_incident_detection.utils.logger import get_logger

logger = get_logger(__name__)


def is_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm.

    Args:
        point: Point to check (x, y).
        polygon: List of polygon vertices [(x1, y1), (x2, y2), ...].

    Returns:
        True if point is inside polygon, False otherwise.
    """
    if not polygon or len(polygon) < 3:
        return True  # No ROI means full frame

    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def is_bbox_in_roi(
    bbox: Tuple[float, float, float, float],
    roi_points: List[Tuple[int, int]],
    min_overlap: float = 0.5,
) -> bool:
    """Check if bounding box overlaps with ROI region.

    Args:
        bbox: Bounding box (x1, y1, x2, y2).
        roi_points: List of ROI polygon vertices.
        min_overlap: Minimum overlap ratio to consider as inside (0.0-1.0).

    Returns:
        True if bbox overlaps with ROI, False otherwise.
    """
    if not roi_points or len(roi_points) < 3:
        return True  # No ROI means full frame

    x1, y1, x2, y2 = bbox
    bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)

    # Check if center point is in ROI
    if is_point_in_polygon(bbox_center, roi_points):
        return True

    # Check if any corner is in ROI
    corners = [
        (x1, y1),  # top-left
        (x2, y1),  # top-right
        (x1, y2),  # bottom-left
        (x2, y2),  # bottom-right
    ]

    corners_inside = sum(1 for corner in corners if is_point_in_polygon(corner, roi_points))

    # If at least min_overlap of corners are inside, consider it as inside
    return corners_inside >= min_overlap * 4


def filter_detections_by_roi(
    detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
    roi_points: List[Tuple[int, int]],
) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
    """Filter detections by ROI region.

    Args:
        detections: List of (class_id, confidence, bbox) detections.
        roi_points: List of ROI polygon vertices.

    Returns:
        Filtered list of detections within ROI.
    """
    if not roi_points or len(roi_points) < 3:
        return detections  # No ROI means return all

    filtered = []
    for class_id, confidence, bbox in detections:
        if is_bbox_in_roi(bbox, roi_points):
            filtered.append((class_id, confidence, bbox))

    return filtered

