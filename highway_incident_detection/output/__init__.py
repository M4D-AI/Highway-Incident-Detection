"""Result output interfaces for downstream integrations."""

from highway_incident_detection.output.builtin_handlers import (
    HTTPWebhookHandler,
    KafkaHandler,
    RedisHandler,
)
from highway_incident_detection.output.result_output import EventResult, ResultHandler

__all__ = [
    "EventResult",
    "HTTPWebhookHandler",
    "KafkaHandler",
    "RedisHandler",
    "ResultHandler",
]
