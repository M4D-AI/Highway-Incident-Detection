"""Example usage of Highway Incident Detection System.

This script demonstrates how to use the system programmatically.
"""

import asyncio
from highway_incident_detection.config.settings import Settings
from highway_incident_detection.models.yolo_model import YOLOModel
from highway_incident_detection.detectors.event_detector import EventDetector
from highway_incident_detection.utils.logger import setup_logger, get_logger


async def example_detection():
    """Example of using the detection components."""
    # Load settings
    settings = Settings()

    # Setup logger
    setup_logger(
        log_dir=settings.logging.log_dir,
        log_level=settings.logging.log_level,
    )
    logger = get_logger(__name__)

    # Initialize YOLO model
    logger.info("Initializing YOLO model...")
    yolo_model = YOLOModel(settings.yolo)

    # Initialize event detector
    logger.info("Initializing event detector...")
    event_detector = EventDetector(settings.event)

    # Example: Process a single image
    # (In real usage, frames come from RTSP streams)
    logger.info("Example: Detection components initialized successfully")
    logger.info("To process video streams, use main.py with RTSP configuration")


if __name__ == "__main__":
    asyncio.run(example_detection())

