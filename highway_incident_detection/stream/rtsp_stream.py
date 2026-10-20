"""RTSP stream processing module with auto-keepalive and reconnection."""

import asyncio
import inspect
from typing import Any, Callable, Dict, Set
from datetime import datetime
import cv2
import numpy as np
from highway_incident_detection.config.settings import StreamConfig
from highway_incident_detection.utils.logger import get_logger

logger = get_logger(__name__)


class RTSPStreamProcessor:
    """Process multiple RTSP streams efficiently with auto-keepalive."""

    def __init__(
        self,
        config: StreamConfig,
        frame_callback: Callable[[str, np.ndarray, int, datetime], None],
    ):
        """Initialize RTSP stream processor.

        Args:
            config: Stream configuration.
            frame_callback: Callback function called for each processed frame.
                           Signature: (stream_name, frame, frame_number, timestamp) -> None
        """
        self.config = config
        self.frame_callback = frame_callback
        self.cap_dict: Dict[str, cv2.VideoCapture] = {}
        self.running = False
        self.tasks: Dict[str, asyncio.Task] = {}
        self.frame_counters: Dict[str, int] = {}
        self.last_frame_time: Dict[str, datetime] = {}
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self.reconnect_attempts: Dict[str, int] = {}
        self.callback_tasks: Dict[str, Set[asyncio.Task]] = {}
        self.dropped_frames: Dict[str, int] = {}

    def _cleanup_capture(self, stream_name: str) -> None:
        """Clean up video capture for a stream.

        Args:
            stream_name: Name of the stream.
        """
        if stream_name in self.cap_dict:
            try:
                cap = self.cap_dict[stream_name]
                if cap.isOpened():
                    cap.release()
                del self.cap_dict[stream_name]
                logger.debug(f"Cleaned up capture for {stream_name}")
            except Exception as e:
                logger.warning(f"Error cleaning up capture for {stream_name}: {e}")

    async def _connect_stream(self, stream_name: str, stream_url: str) -> bool:
        """Connect to RTSP stream.

        Args:
            stream_name: Name of the stream.
            stream_url: RTSP URL.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            logger.info(f"Connecting to stream {stream_name}: {stream_url}")

            # Clean up existing connection if any
            self._cleanup_capture(stream_name)

            # Create new connection
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            cap.set(cv2.CAP_PROP_FPS, 25)  # Set expected FPS

            # Try to read a frame to verify connection
            ret, _ = await asyncio.to_thread(cap.read)
            if not ret or not cap.isOpened():
                cap.release()
                logger.warning(f"Failed to verify connection to {stream_name}")
                return False

            self.cap_dict[stream_name] = cap
            self.reconnect_attempts[stream_name] = 0
            logger.info(f"Successfully connected to stream {stream_name}")
            return True

        except Exception as e:
            logger.error(f"Error connecting to stream {stream_name}: {e}")
            self._cleanup_capture(stream_name)
            return False

    async def _heartbeat_check(self, stream_name: str, stream_url: str) -> None:
        """Periodic heartbeat check to ensure stream is alive.

        Args:
            stream_name: Name of the stream.
            stream_url: RTSP URL.
        """
        while self.running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                if not self.running:
                    break

                # Check if stream is still alive
                if stream_name not in self.cap_dict:
                    continue

                # Check last frame time
                if stream_name in self.last_frame_time:
                    time_since_last_frame = (
                        datetime.now() - self.last_frame_time[stream_name]
                    ).total_seconds()

                    if time_since_last_frame > self.config.no_frame_timeout:
                        logger.warning(
                            f"No frames received from {stream_name} for "
                            f"{time_since_last_frame:.1f} seconds, reconnecting..."
                        )
                        # Trigger reconnection
                        self._cleanup_capture(stream_name)
                        await asyncio.sleep(1)

                # Verify connection is still open
                if stream_name in self.cap_dict:
                    cap = self.cap_dict[stream_name]
                    if not cap.isOpened():
                        logger.warning(
                            f"Connection to {stream_name} is closed, reconnecting..."
                        )
                        self._cleanup_capture(stream_name)

            except Exception as e:
                logger.error(f"Error in heartbeat check for {stream_name}: {e}")

    def _track_callback_task(self, stream_name: str, task: asyncio.Task) -> None:
        """Track an in-flight frame callback task."""
        pending_tasks = self.callback_tasks.setdefault(stream_name, set())
        pending_tasks.add(task)

        def _cleanup(done_task: asyncio.Task) -> None:
            pending_tasks.discard(done_task)
            try:
                done_task.result()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Unhandled frame callback error for {stream_name}: {e}")

        task.add_done_callback(_cleanup)

    async def _dispatch_frame_callback(
        self,
        stream_name: str,
        frame: np.ndarray,
        frame_number: int,
        timestamp: datetime,
    ) -> None:
        """Dispatch frame processing without blocking frame reads."""
        callback_result = self.frame_callback(
            stream_name,
            frame,
            frame_number,
            timestamp,
        )
        if inspect.isawaitable(callback_result):
            task = asyncio.create_task(callback_result)
            self._track_callback_task(stream_name, task)

    async def _wait_for_callback_slot(self, stream_name: str) -> bool:
        """Reserve a processing slot or apply backpressure policy."""
        pending_tasks = self.callback_tasks.setdefault(stream_name, set())
        if len(pending_tasks) < self.config.max_pending_frames:
            return True

        if self.config.drop_frames_when_busy:
            dropped = self.dropped_frames.get(stream_name, 0) + 1
            self.dropped_frames[stream_name] = dropped
            if dropped == 1 or dropped % 50 == 0:
                logger.warning(
                    f"Stream {stream_name} is saturated, dropped {dropped} frames"
                )
            return False

        done, _ = await asyncio.wait(
            pending_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            pending_tasks.discard(task)
        return True

    async def _process_stream(self, stream_name: str, stream_url: str) -> None:
        """Process a single RTSP stream with auto-reconnection.

        Args:
            stream_name: Name of the stream.
            stream_url: RTSP URL of the stream.
        """
        frame_counter = 0
        consecutive_read_failures = 0
        max_read_failures = 10

        while self.running:
            try:
                # Connect if not connected
                if stream_name not in self.cap_dict or not self.cap_dict[stream_name].isOpened():
                    if not self.config.auto_reconnect:
                        logger.error(f"Auto-reconnect disabled, stopping {stream_name}")
                        break

                    # Check reconnect attempts limit
                    if (
                        self.config.max_reconnect_attempts > 0
                        and self.reconnect_attempts.get(stream_name, 0) >= self.config.max_reconnect_attempts
                    ):
                        logger.error(
                            f"Max reconnection attempts ({self.config.max_reconnect_attempts}) "
                            f"reached for {stream_name}. Stopping stream."
                        )
                        break

                    # Attempt connection
                    self.reconnect_attempts[stream_name] = (
                        self.reconnect_attempts.get(stream_name, 0) + 1
                    )
                    connected = await self._connect_stream(stream_name, stream_url)

                    if not connected:
                        wait_time = min(
                            self.config.reconnect_interval * self.reconnect_attempts[stream_name],
                            60,  # Max 60 seconds
                        )
                        logger.info(
                            f"Reconnection attempt {self.reconnect_attempts[stream_name]} "
                            f"failed for {stream_name}, retrying in {wait_time} seconds..."
                        )
                        await asyncio.sleep(wait_time)
                        continue

                cap = self.cap_dict[stream_name]

                # Read frame with timeout simulation
                ret, frame = await asyncio.to_thread(cap.read)

                if not ret or frame is None:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= max_read_failures:
                        logger.warning(
                            f"Multiple read failures for {stream_name}, "
                            f"reconnecting..."
                        )
                        self._cleanup_capture(stream_name)
                        consecutive_read_failures = 0
                        await asyncio.sleep(self.config.reconnect_interval)
                    else:
                        await asyncio.sleep(0.1)
                    continue

                # Reset failure counter on success
                consecutive_read_failures = 0

                # Frame skipping for efficiency
                frame_counter += 1
                if frame_counter % self.config.frame_skip != 0:
                    continue

                # Process frame
                timestamp = datetime.now()
                self.last_frame_time[stream_name] = timestamp

                has_capacity = await self._wait_for_callback_slot(stream_name)
                if not has_capacity:
                    continue

                # Number frames only after they are accepted, so dropped frames
                # never leave gaps in the per-stream sequence.
                frame_number = self.frame_counters.get(stream_name, 0) + 1
                self.frame_counters[stream_name] = frame_number

                # Call callback (this should be fast, non-blocking)
                try:
                    await self._dispatch_frame_callback(
                        stream_name,
                        frame,
                        frame_number,
                        timestamp,
                    )
                except Exception as e:
                    logger.error(f"Error in frame callback for {stream_name}: {e}")

                # Small delay to prevent CPU overload
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error processing stream {stream_name}: {e}")
                consecutive_read_failures += 1

                # Clean up failed connection
                self._cleanup_capture(stream_name)

                if self.config.auto_reconnect:
                    wait_time = min(
                        self.config.reconnect_interval * self.reconnect_attempts.get(stream_name, 1),
                        60,  # Max 60 seconds
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Auto-reconnect disabled, stopping {stream_name}")
                    break

        # Cleanup
        self._cleanup_capture(stream_name)
        logger.info(f"Stream processor stopped for {stream_name}")

    async def start(self) -> None:
        """Start processing all enabled streams."""
        self.running = True

        for stream_config in self.config.streams:
            if not stream_config.get("enabled", False):
                logger.info(f"Stream {stream_config['name']} is disabled, skipping")
                continue

            stream_name = stream_config["name"]
            stream_url = stream_config["url"]

            # Initialize counters
            self.frame_counters[stream_name] = 0
            self.reconnect_attempts[stream_name] = 0
            self.last_frame_time[stream_name] = datetime.now()
            self.callback_tasks[stream_name] = set()
            self.dropped_frames[stream_name] = 0

            # Create task for stream processing
            task = asyncio.create_task(
                self._process_stream(stream_name, stream_url)
            )
            self.tasks[stream_name] = task

            # Create heartbeat task
            heartbeat_task = asyncio.create_task(
                self._heartbeat_check(stream_name, stream_url)
            )
            self.heartbeat_tasks[stream_name] = heartbeat_task

            logger.info(f"Started processing stream: {stream_name}")

        # Wait for all tasks
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)

        callback_tasks = [
            task
            for stream_tasks in self.callback_tasks.values()
            for task in stream_tasks
        ]
        if callback_tasks:
            await asyncio.gather(*callback_tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Stop processing all streams."""
        logger.info("Stopping all stream processors...")
        self.running = False

        # Cancel heartbeat tasks
        for task in self.heartbeat_tasks.values():
            task.cancel()
        if self.heartbeat_tasks:
            await asyncio.gather(*self.heartbeat_tasks.values(), return_exceptions=True)

        # Wait for stream tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)

        callback_tasks = [
            task
            for stream_tasks in self.callback_tasks.values()
            for task in stream_tasks
        ]
        if callback_tasks:
            await asyncio.gather(*callback_tasks, return_exceptions=True)

        # Release all captures
        for stream_name in list(self.cap_dict.keys()):
            self._cleanup_capture(stream_name)

        self.cap_dict.clear()
        self.tasks.clear()
        self.heartbeat_tasks.clear()
        self.callback_tasks.clear()
        logger.info("All stream processors stopped")

    def get_stream_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all streams.

        Returns:
            Dictionary with stream status information.
        """
        status = {}
        for stream_name in self.frame_counters.keys():
            status[stream_name] = {
                "frame_count": self.frame_counters.get(stream_name, 0),
                "is_connected": (
                    stream_name in self.cap_dict
                    and self.cap_dict[stream_name].isOpened()
                ),
                "reconnect_attempts": self.reconnect_attempts.get(stream_name, 0),
                "last_frame_time": (
                    self.last_frame_time[stream_name].isoformat()
                    if stream_name in self.last_frame_time
                    else None
                ),
                "pending_frames": len(self.callback_tasks.get(stream_name, set())),
                "dropped_frames": self.dropped_frames.get(stream_name, 0),
            }
        return status
