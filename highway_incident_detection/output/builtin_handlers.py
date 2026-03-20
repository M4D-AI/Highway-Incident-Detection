"""Built-in downstream handlers for HTTP, Kafka, and Redis outputs."""

import asyncio
import json
from typing import Any, Dict, Optional
from urllib import request

from highway_incident_detection.output.result_output import EventResult


class HTTPWebhookHandler:
    """Post event results to an HTTP webhook endpoint."""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 10,
    ):
        self.url = url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds

    async def __call__(self, result: EventResult) -> None:
        await asyncio.to_thread(self._post, result.to_dict())

    def _post(self, payload: Dict[str, Any]) -> None:
        req = request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **self.headers,
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds):
            return None


class KafkaHandler:
    """Publish event results to a Kafka topic."""

    def __init__(
        self,
        topic: str,
        producer: Any = None,
        bootstrap_servers: Optional[str] = None,
        key_field: str = "event_id",
    ):
        self.topic = topic
        self.key_field = key_field
        self.producer = producer or self._build_producer(bootstrap_servers)

    async def __call__(self, result: EventResult) -> None:
        await asyncio.to_thread(self._send, result.to_dict())

    def _send(self, payload: Dict[str, Any]) -> None:
        key_value = payload.get(self.key_field)
        message_key = str(key_value).encode("utf-8") if key_value is not None else None
        future = self.producer.send(self.topic, value=payload, key=message_key)
        if hasattr(future, "get"):
            future.get(timeout=10)
        if hasattr(self.producer, "flush"):
            self.producer.flush()

    def _build_producer(self, bootstrap_servers: Optional[str]) -> Any:
        if not bootstrap_servers:
            raise ValueError("bootstrap_servers is required when producer is not provided")

        try:
            from kafka import KafkaProducer
        except ImportError as exc:
            raise RuntimeError(
                "KafkaHandler requires kafka-python. Install it or provide a producer instance."
            ) from exc

        return KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )


class RedisHandler:
    """Publish event results to Redis using list push, channel publish, or both."""

    def __init__(
        self,
        client: Any = None,
        url: Optional[str] = None,
        list_key: Optional[str] = "highway_incident_results",
        channel: Optional[str] = None,
    ):
        self.list_key = list_key
        self.channel = channel
        self.client = client or self._build_client(url)

    async def __call__(self, result: EventResult) -> None:
        await asyncio.to_thread(self._publish, result.to_dict())

    def _publish(self, payload: Dict[str, Any]) -> None:
        message = json.dumps(payload, ensure_ascii=False)
        if self.list_key:
            self.client.rpush(self.list_key, message)
        if self.channel:
            self.client.publish(self.channel, message)

    def _build_client(self, url: Optional[str]) -> Any:
        if not url:
            raise ValueError("url is required when client is not provided")

        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "RedisHandler requires redis-py. Install it or provide a client instance."
            ) from exc

        return redis.from_url(url)
