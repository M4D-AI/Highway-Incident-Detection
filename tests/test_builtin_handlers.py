import asyncio
import json
from datetime import datetime

from highway_incident_detection import (
    HTTPWebhookHandler,
    KafkaHandler,
    RedisHandler,
)
from highway_incident_detection.detectors.event_types import EventType, IncidentEvent
from highway_incident_detection.output.result_output import EventResult


class _FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeKafkaFuture:
    def __init__(self):
        self.timeout = None

    def get(self, timeout=None):
        self.timeout = timeout
        return None


class _FakeKafkaProducer:
    def __init__(self):
        self.messages = []
        self.flush_called = False
        self.future = _FakeKafkaFuture()

    def send(self, topic, value=None, key=None):
        self.messages.append({"topic": topic, "value": value, "key": key})
        return self.future

    def flush(self):
        self.flush_called = True


class _FakeRedisClient:
    def __init__(self):
        self.list_messages = []
        self.channel_messages = []

    def rpush(self, key, value):
        self.list_messages.append((key, value))

    def publish(self, channel, value):
        self.channel_messages.append((channel, value))


def _build_result() -> EventResult:
    event = IncidentEvent(
        event_type=EventType.ROAD_DEBRIS,
        stream_name="cam_1",
        timestamp=datetime.now(),
        frame_number=8,
        confidence=0.93,
        description="Road debris detected",
        metadata={"lane": "fast"},
    )
    return EventResult(
        event=event,
        event_id="evt-123",
        keyframe_path="key.jpg",
        video_path="clip.mp4",
        summary_path="summary.json",
        audit_path="audit.json",
        audit={"confirmed": True, "risk_level": "high"},
    )


def test_http_webhook_handler_posts_json(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse()

    monkeypatch.setattr("highway_incident_detection.output.builtin_handlers.request.urlopen", fake_urlopen)

    handler = HTTPWebhookHandler(
        "http://127.0.0.1:8080/webhook",
        headers={"X-Test": "1"},
        timeout_seconds=5,
    )
    asyncio.run(handler(_build_result()))

    assert captured["url"] == "http://127.0.0.1:8080/webhook"
    assert captured["timeout"] == 5
    assert captured["payload"]["event_id"] == "evt-123"
    assert captured["headers"]["X-test"] == "1"


def test_kafka_handler_uses_provided_producer():
    producer = _FakeKafkaProducer()
    handler = KafkaHandler(topic="incident-events", producer=producer)

    asyncio.run(handler(_build_result()))

    assert len(producer.messages) == 1
    assert producer.messages[0]["topic"] == "incident-events"
    assert producer.messages[0]["value"]["event_id"] == "evt-123"
    assert producer.messages[0]["key"] == b"evt-123"
    assert producer.flush_called is True
    assert producer.future.timeout == 10


def test_redis_handler_uses_provided_client():
    client = _FakeRedisClient()
    handler = RedisHandler(
        client=client,
        list_key="incident-list",
        channel="incident-channel",
    )

    asyncio.run(handler(_build_result()))

    assert len(client.list_messages) == 1
    assert len(client.channel_messages) == 1
    assert client.list_messages[0][0] == "incident-list"
    assert client.channel_messages[0][0] == "incident-channel"
    assert json.loads(client.list_messages[0][1])["event_id"] == "evt-123"
