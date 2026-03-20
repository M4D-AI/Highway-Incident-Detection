"""Example usage of Highway Incident Detection System.

This script demonstrates how to use the system programmatically.
"""

import asyncio
from highway_incident_detection import (
    EventResult,
    HTTPWebhookHandler,
    HighwayIncidentDetectionSystem,
)


async def handle_result(result: EventResult) -> None:
    """Custom downstream processing hook."""
    print("received result:", result.to_dict())


async def example_detection():
    """Example of subscribing to output results."""
    system = HighwayIncidentDetectionSystem(config_path="config/config.yaml")
    system.register_result_handler(handle_result)
    system.register_result_handler(
        HTTPWebhookHandler("http://127.0.0.1:8080/webhook")
    )
    await system.run()


if __name__ == "__main__":
    asyncio.run(example_detection())
