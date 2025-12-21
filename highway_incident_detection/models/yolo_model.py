"""YOLO model wrapper for object detection with auto-restart capability."""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from datetime import datetime, timedelta
from ultralytics import YOLO
from highway_incident_detection.config.settings import YOLOConfig
from highway_incident_detection.utils.logger import get_logger

logger = get_logger(__name__)


class YOLOModel:
    """YOLO model wrapper for efficient inference with auto-restart."""

    def __init__(self, config: YOLOConfig):
        """Initialize YOLO model.

        Args:
            config: YOLO configuration.
        """
        self.config = config
        self.model: Optional[YOLO] = None
        self._load_model()

        # Health monitoring
        self.last_successful_inference: Optional[datetime] = None
        self.consecutive_failures: int = 0
        self.total_inferences: int = 0
        self.total_failures: int = 0
        self.last_health_check: Optional[datetime] = None

    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            logger.info(f"Loading YOLO model from {self.config.model_path}")
            if self.model is not None:
                # Clean up old model
                try:
                    del self.model
                except Exception:
                    pass
                import gc
                gc.collect()

            self.model = YOLO(self.config.model_path)
            logger.info(f"YOLO model loaded successfully on {self.config.device}")
            self.consecutive_failures = 0
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            if self.config.auto_restart:
                logger.warning("Auto-restart enabled, will retry on next inference")
            else:
                raise

    def _check_health(self) -> bool:
        """Check model health and restart if needed.

        Returns:
            True if model is healthy, False otherwise.
        """
        if not self.config.auto_restart:
            return self.model is not None

        now = datetime.now()

        # Check if health check is needed
        if (
            self.last_health_check is None
            or (now - self.last_health_check).total_seconds() >= self.config.health_check_interval
        ):
            self.last_health_check = now

            # Check for too many consecutive failures
            if self.consecutive_failures >= self.config.max_failures:
                logger.warning(
                    f"Model has {self.consecutive_failures} consecutive failures, "
                    f"attempting restart..."
                )
                try:
                    self._load_model()
                    return self.model is not None
                except Exception as e:
                    logger.error(f"Failed to restart model: {e}")
                    return False

        return self.model is not None

    def predict(
        self, image: np.ndarray, verbose: bool = False
    ) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
        """Run inference on image.

        Args:
            image: Input image as numpy array (BGR format).
            verbose: Whether to print verbose output.

        Returns:
            List of (class_id, confidence, bbox) tuples where bbox is (x1, y1, x2, y2).
        """
        # Check health and restart if needed
        if not self._check_health():
            if self.config.auto_restart:
                logger.warning("Model unhealthy, attempting restart...")
                try:
                    self._load_model()
                except Exception as e:
                    logger.error(f"Failed to restart model: {e}")
                    return []

        if self.model is None:
            if self.config.auto_restart:
                logger.warning("Model not loaded, attempting to load...")
                try:
                    self._load_model()
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    return []
            else:
                raise RuntimeError("Model not loaded")

        try:
            self.total_inferences += 1
            results = self.model.predict(
                image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                device=self.config.device,
                imgsz=self.config.img_size,
                verbose=verbose,
            )

            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    confidence = float(boxes.conf[i].item())
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    detections.append((class_id, confidence, (float(x1), float(y1), float(x2), float(y2))))

            # Reset failure counter on success
            self.consecutive_failures = 0
            self.last_successful_inference = datetime.now()

            return detections
        except Exception as e:
            self.total_failures += 1
            self.consecutive_failures += 1
            logger.error(f"Prediction failed (consecutive failures: {self.consecutive_failures}): {e}")

            # Attempt restart if auto-restart is enabled
            if self.config.auto_restart and self.consecutive_failures >= self.config.max_failures:
                logger.warning("Too many failures, attempting model restart...")
                try:
                    self._load_model()
                except Exception as restart_error:
                    logger.error(f"Failed to restart model: {restart_error}")

            return []

    def predict_batch(
        self, images: List[np.ndarray], verbose: bool = False
    ) -> List[List[Tuple[int, float, Tuple[float, float, float, float]]]]:
        """Run batch inference on multiple images.

        Args:
            images: List of input images as numpy arrays (BGR format).
            verbose: Whether to print verbose output.

        Returns:
            List of detection lists, one per image.
        """
        if self.model is None:
            if self.config.auto_restart:
                try:
                    self._load_model()
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    return [[] for _ in images]
            else:
                raise RuntimeError("Model not loaded")

        try:
            self.total_inferences += len(images)
            results = self.model.predict(
                images,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                device=self.config.device,
                imgsz=self.config.img_size,
                verbose=verbose,
            )

            all_detections = []
            for result in results:
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        class_id = int(boxes.cls[i].item())
                        confidence = float(boxes.conf[i].item())
                        bbox = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        detections.append(
                            (class_id, confidence, (float(x1), float(y1), float(x2), float(y2)))
                        )
                all_detections.append(detections)

            # Reset failure counter on success
            self.consecutive_failures = 0
            self.last_successful_inference = datetime.now()

            return all_detections
        except Exception as e:
            self.total_failures += len(images)
            self.consecutive_failures += 1
            logger.error(f"Batch prediction failed: {e}")

            # Attempt restart if auto-restart is enabled
            if self.config.auto_restart and self.consecutive_failures >= self.config.max_failures:
                logger.warning("Too many failures, attempting model restart...")
                try:
                    self._load_model()
                except Exception as restart_error:
                    logger.error(f"Failed to restart model: {restart_error}")

            return [[] for _ in images]

    def get_stats(self) -> dict:
        """Get model statistics.

        Returns:
            Dictionary with model statistics.
        """
        return {
            "total_inferences": self.total_inferences,
            "total_failures": self.total_failures,
            "consecutive_failures": self.consecutive_failures,
            "last_successful_inference": (
                self.last_successful_inference.isoformat()
                if self.last_successful_inference
                else None
            ),
            "is_loaded": self.model is not None,
        }
