"""Logging utility module."""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

_logger_configured: bool = False


def setup_logger(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    log_rotation: str = "100 MB",
    log_retention: str = "30 days",
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
) -> None:
    """Configure loguru logger.

    Args:
        log_dir: Directory for log files.
        log_level: Logging level (DEBUG/INFO/WARNING/ERROR).
        log_rotation: Log file rotation size.
        log_retention: Log file retention period.
        log_format: Log format string.
    """
    global _logger_configured

    if _logger_configured:
        return

    # Remove default handler
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Add file handler for all logs
    logger.add(
        log_path / "highway_incident_{time:YYYY-MM-DD}.log",
        format=log_format,
        level=log_level,
        rotation=log_rotation,
        retention=log_retention,
        compression="zip",
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
    )

    # Add separate file handler for errors
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="ERROR",
        rotation=log_rotation,
        retention=log_retention,
        compression="zip",
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
    )

    _logger_configured = True
    logger.info("Logger configured successfully")


def get_logger(name: Optional[str] = None):
    """Get logger instance.

    Args:
        name: Optional logger name.

    Returns:
        Logger instance.
    """
    if name:
        return logger.bind(name=name)
    return logger

