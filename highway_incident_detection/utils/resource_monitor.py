"""Resource monitoring and health check utilities."""

import time
import psutil
import gc
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from highway_incident_detection.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceStats:
    """Resource statistics."""

    memory_mb: float
    memory_percent: float
    cpu_percent: float
    timestamp: datetime


class ResourceMonitor:
    """Monitor system resources and detect anomalies."""

    def __init__(
        self,
        memory_threshold_mb: float = 4096.0,
        memory_threshold_percent: float = 80.0,
        cpu_threshold_percent: float = 90.0,
    ):
        """Initialize resource monitor.

        Args:
            memory_threshold_mb: Memory threshold in MB.
            memory_threshold_percent: Memory usage percentage threshold.
            cpu_threshold_percent: CPU usage percentage threshold.
        """
        self.memory_threshold_mb = memory_threshold_mb
        self.memory_threshold_percent = memory_threshold_percent
        self.cpu_threshold_percent = cpu_threshold_percent
        self.process = psutil.Process()

    def get_stats(self) -> ResourceStats:
        """Get current resource statistics.

        Returns:
            Resource statistics.
        """
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        cpu_percent = self.process.cpu_percent(interval=0.1)

        return ResourceStats(
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            cpu_percent=cpu_percent,
            timestamp=datetime.now(),
        )

    def check_health(self):
        """Check if system is healthy.

        Returns:
            Tuple of (is_healthy, message).
        """
        stats = self.get_stats()

        issues = []
        if stats.memory_mb > self.memory_threshold_mb:
            issues.append(
                f"Memory usage too high: {stats.memory_mb:.2f} MB "
                f"(threshold: {self.memory_threshold_mb} MB)"
            )
        if stats.memory_percent > self.memory_threshold_percent:
            issues.append(
                f"Memory usage percentage too high: {stats.memory_percent:.2f}% "
                f"(threshold: {self.memory_threshold_percent}%)"
            )
        if stats.cpu_percent > self.cpu_threshold_percent:
            issues.append(
                f"CPU usage too high: {stats.cpu_percent:.2f}% "
                f"(threshold: {self.cpu_threshold_percent}%)"
            )

        if issues:
            return False, "; ".join(issues)

        return True, "System healthy"

    def force_gc(self) -> None:
        """Force garbage collection."""
        collected = gc.collect()
        logger.debug(f"Garbage collection: {collected} objects collected")

