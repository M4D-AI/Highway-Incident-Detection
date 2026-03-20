from datetime import datetime
from types import SimpleNamespace

import numpy as np

from highway_incident_detection.models.yolo_model import YOLOModel


class _FakeResult:
    boxes = None


class _FakeBackend:
    def predict(self, images, **kwargs):
        return [_FakeResult() for _ in images]


def test_predict_batch_rechecks_health_before_inference():
    model = YOLOModel.__new__(YOLOModel)
    model.config = SimpleNamespace(
        auto_restart=True,
        confidence_threshold=0.5,
        iou_threshold=0.45,
        device="cpu",
        img_size=640,
        max_failures=5,
        health_check_interval=300,
    )
    model.model = _FakeBackend()
    model.last_successful_inference = None
    model.consecutive_failures = 0
    model.total_inferences = 0
    model.total_failures = 0
    model.last_health_check = None

    calls = {"check": 0, "load": 0}

    def check_health():
        calls["check"] += 1
        return False

    def load_model():
        calls["load"] += 1
        model.model = _FakeBackend()

    model._check_health = check_health
    model._load_model = load_model

    results = model.predict_batch([np.zeros((4, 4, 3), dtype=np.uint8)])

    assert calls["check"] == 1
    assert calls["load"] == 1
    assert results == [[]]
    assert model.total_inferences == 1
    assert isinstance(model.last_successful_inference, datetime)
