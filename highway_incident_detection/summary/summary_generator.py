"""Asynchronous summary generation module."""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import cv2
import numpy as np
from collections import deque
import aiofiles
from highway_incident_detection.detectors.event_types import IncidentEvent
from highway_incident_detection.config.settings import OutputConfig
from highway_incident_detection.utils.logger import get_logger

logger = get_logger(__name__)


class SummaryGenerator:
    """Generate summaries, keyframes, and videos asynchronously."""

    def __init__(self, config: OutputConfig):
        """Initialize summary generator.

        Args:
            config: Output configuration.
        """
        self.config = config
        self._create_directories()

        # Frame buffers for video generation (stream_name -> deque of frames)
        self.frame_buffers: Dict[str, deque] = {}
        self.frame_timestamps: Dict[str, deque] = {}
        self.frame_numbers: Dict[str, deque] = {}

        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.running = False

    def _create_directories(self) -> None:
        """Create output directories."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.keyframe_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.video_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.summary_dir).mkdir(parents=True, exist_ok=True)

    def add_frame(
        self,
        stream_name: str,
        frame: np.ndarray,
        frame_number: int,
        timestamp: datetime,
    ) -> None:
        """Add frame to buffer for video generation.

        Args:
            stream_name: Name of the stream.
            frame: Frame image.
            frame_number: Frame number.
            timestamp: Frame timestamp.
        """
        if stream_name not in self.frame_buffers:
            # Calculate buffer size based on video duration and FPS
            buffer_size = self.config.video_duration_seconds * self.config.video_fps
            self.frame_buffers[stream_name] = deque(maxlen=buffer_size)
            self.frame_timestamps[stream_name] = deque(maxlen=buffer_size)
            self.frame_numbers[stream_name] = deque(maxlen=buffer_size)

        self.frame_buffers[stream_name].append(frame.copy())
        self.frame_timestamps[stream_name].append(timestamp)
        self.frame_numbers[stream_name].append(frame_number)

    async def generate_keyframe(
        self, event: IncidentEvent, frame: np.ndarray
    ) -> str:
        """Generate keyframe image for event.

        Args:
            event: Incident event.
            frame: Frame image.

        Returns:
            Path to saved keyframe image.
        """
        try:
            # Create filename
            timestamp_str = event.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{event.stream_name}_{event.event_type.value}_{timestamp_str}_"
                f"frame{event.frame_number}.jpg"
            )
            filepath = Path(self.config.keyframe_dir) / filename

            # Draw bounding box if available
            output_frame = frame.copy()
            if event.bbox:
                x1, y1, x2, y2 = map(int, event.bbox)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    output_frame,
                    f"{event.event_type.value} ({event.confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Save image
            cv2.imwrite(str(filepath), output_frame)
            logger.info(f"Keyframe saved: {filepath}")

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to generate keyframe: {e}")
            return ""

    async def generate_video(
        self, event: IncidentEvent, frames: List[np.ndarray]
    ) -> str:
        """Generate video clip for event.

        Args:
            event: Incident event.
            frames: List of frames for the video.

        Returns:
            Path to saved video file.
        """
        if not frames:
            logger.warning("No frames provided for video generation")
            return ""

        try:
            # Create filename
            timestamp_str = event.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{event.stream_name}_{event.event_type.value}_{timestamp_str}_"
                f"frame{event.frame_number}.mp4"
            )
            filepath = Path(self.config.video_dir) / filename

            # Get frame dimensions
            height, width = frames[0].shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(filepath),
                fourcc,
                self.config.video_fps,
                (width, height),
            )

            # Write frames
            for frame in frames:
                out.write(frame)

            out.release()
            logger.info(f"Video saved: {filepath}")

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to generate video: {e}")
            return ""

    async def generate_summary(self, event: IncidentEvent, keyframe_path: str, video_path: str) -> str:
        """Generate summary JSON file for event.

        Args:
            event: Incident event.
            keyframe_path: Path to keyframe image.
            video_path: Path to video file.

        Returns:
            Path to saved summary file.
        """
        try:
            # Create summary data
            summary = {
                "event_id": f"{event.stream_name}_{event.event_type.value}_{event.timestamp.isoformat()}",
                "event_type": event.event_type.value,
                "stream_name": event.stream_name,
                "timestamp": event.timestamp.isoformat(),
                "frame_number": event.frame_number,
                "confidence": event.confidence,
                "description": event.description,
                "bbox": list(event.bbox) if event.bbox else None,
                "metadata": event.metadata,
                "keyframe_path": keyframe_path,
                "video_path": video_path,
            }

            # Create filename
            timestamp_str = event.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{event.stream_name}_{event.event_type.value}_{timestamp_str}_"
                f"frame{event.frame_number}.json"
            )
            filepath = Path(self.config.summary_dir) / filename

            # Save JSON
            async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                await f.write(json.dumps(summary, indent=2, ensure_ascii=False))

            logger.info(f"Summary saved: {filepath}")

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return ""

    async def process_event(self, event: IncidentEvent) -> None:
        """Process event asynchronously (generate keyframe, video, summary).

        Args:
            event: Incident event.
        """
        try:
            # Get frames from buffer
            stream_name = event.stream_name
            if stream_name not in self.frame_buffers:
                logger.warning(f"No frame buffer found for stream {stream_name}")
                return

            frames = list(self.frame_buffers[stream_name])
            if not frames:
                logger.warning(f"No frames in buffer for stream {stream_name}")
                return

            # Get the frame at event time (or closest)
            event_frame = None
            event_frame_idx = -1
            for i, frame_num in enumerate(self.frame_numbers[stream_name]):
                if frame_num == event.frame_number:
                    event_frame = frames[i]
                    event_frame_idx = i
                    break

            if event_frame is None and frames:
                # Use last frame if exact match not found
                event_frame = frames[-1]
                event_frame_idx = len(frames) - 1

            # Generate keyframe
            keyframe_path = await self.generate_keyframe(event, event_frame)

            # Generate video (use frames around event)
            video_frames = []
            if event_frame_idx >= 0:
                # Get frames around the event (before and after)
                start_idx = max(0, event_frame_idx - self.config.video_fps * 5)
                end_idx = min(len(frames), event_frame_idx + self.config.video_fps * 5)
                video_frames = frames[start_idx:end_idx]

            if not video_frames:
                video_frames = frames[-self.config.video_fps * self.config.video_duration_seconds:]

            video_path = await self.generate_video(event, video_frames)

            # Generate summary
            await self.generate_summary(event, keyframe_path, video_path)

        except Exception as e:
            logger.error(f"Error processing event: {e}")

    async def _process_event_queue(self) -> None:
        """Process events from queue asynchronously."""
        while self.running:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                await self.process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event queue processing: {e}")

    async def add_event(self, event: IncidentEvent) -> None:
        """Add event to processing queue (non-blocking).

        Args:
            event: Incident event.
        """
        await self.event_queue.put(event)

    async def start(self) -> None:
        """Start summary generator."""
        self.running = True
        self.processing_task = asyncio.create_task(self._process_event_queue())
        logger.info("Summary generator started")

    async def stop(self) -> None:
        """Stop summary generator."""
        self.running = False
        if self.processing_task:
            await self.processing_task
        logger.info("Summary generator stopped")

