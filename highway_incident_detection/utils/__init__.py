"""Utility functions module."""

from highway_incident_detection.utils.logger import setup_logger, get_logger
from highway_incident_detection.utils.tracker import ObjectTracker
from highway_incident_detection.utils.resource_monitor import ResourceMonitor, ResourceStats
from highway_incident_detection.utils.roi_filter import (
    filter_detections_by_roi,
    is_bbox_in_roi,
    is_point_in_polygon,
)

__all__ = [
    "setup_logger",
    "get_logger",
    "ObjectTracker",
    "ResourceMonitor",
    "ResourceStats",
    "filter_detections_by_roi",
    "is_bbox_in_roi",
    "is_point_in_polygon",
]

