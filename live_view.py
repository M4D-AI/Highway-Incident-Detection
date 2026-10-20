"""Live viewer: run the detection system and show annotated frames in a window.

Usage (from the repo root):
    python live_view.py --config config/local_gpu.yaml

Press q or Esc in the window (or Ctrl+C) to quit.
"""

import argparse
import asyncio
import threading
import time
from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np

from highway_incident_detection.system import HighwayIncidentDetectionSystem

# Zone outline colors (BGR), cycled per zone
ZONE_COLORS = [(255, 200, 0), (0, 200, 255), (200, 0, 255), (0, 255, 120)]
EVENT_SHOW_SECONDS = 4.0


class LiveViewSystem(HighwayIncidentDetectionSystem):
    """Detection system that also keeps the latest annotated frame per stream."""

    def __init__(self, config_path: str):
        super().__init__(config_path=config_path)
        self.class_names: Dict[int, str] = {}
        self.view_lock = threading.Lock()
        self.latest: Dict[str, np.ndarray] = {}
        self.fps: Dict[str, deque] = {}
        self.recent_events: Dict[str, deque] = {}
        self.event_counts: Dict[str, int] = {}

    def _finalize_frame_locked(self, stream_name, frame, frame_number, timestamp, detections):
        started = time.perf_counter()
        events = super()._finalize_frame_locked(
            stream_name, frame, frame_number, timestamp, detections
        )
        self._annotate(stream_name, frame, frame_number, detections, events)
        self.fps.setdefault(stream_name, deque(maxlen=30)).append(time.perf_counter() - started)
        return events

    def _annotate(self, stream_name, frame, frame_number, detections, events) -> None:
        if not self.class_names and self.yolo_model.model is not None:
            self.class_names = dict(self.yolo_model.model.names)

        view = frame.copy()
        height, width = view.shape[:2]
        scale = max(1.0, width / 960)  # keep text readable on 1080p

        lane_config = self.lane_configs.get(stream_name, {})
        if lane_config.get("enabled") and lane_config.get("zones"):
            zones = self._get_scaled_zones(stream_name, lane_config, width, height)
            for index, zone in enumerate(zones):
                points = np.array(zone["roi_points"], dtype=np.int32)
                if len(points) >= 3:
                    color = ZONE_COLORS[index % len(ZONE_COLORS)]
                    cv2.polylines(view, [points], True, color, max(1, int(2 * scale)))
                    x, y = points.min(axis=0)
                    cv2.putText(view, zone["name"], (int(x) + 4, int(y) + int(16 * scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, max(1, int(scale)))

        for class_id, confidence, (x1, y1, x2, y2) in detections:
            name = self.class_names.get(int(class_id), str(class_id))
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                          max(1, int(scale)))
            cv2.putText(view, f"{name} {confidence:.2f}", (int(x1), max(0, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (0, 255, 0), max(1, int(scale)))

        now = time.monotonic()
        recent = self.recent_events.setdefault(stream_name, deque(maxlen=6))
        for event in events:
            recent.append((now, event))
            self.event_counts[stream_name] = self.event_counts.get(stream_name, 0) + 1

        for shown_at, event in recent:
            if now - shown_at > EVENT_SHOW_SECONDS or not event.bbox:
                continue
            x1, y1, x2, y2 = event.bbox
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255),
                          max(2, int(3 * scale)))

        lines = [f"{stream_name}  frame {frame_number}  {width}x{height}"]
        lines.append(f"detections {len(detections)}  events {self.event_counts.get(stream_name, 0)}")
        lines += [
            f"EVENT {event.event_type_value} ({event.metadata.get('lane_zone', '-')}) "
            f"{event.confidence:.2f}"
            for shown_at, event in recent
            if now - shown_at <= EVENT_SHOW_SECONDS
        ]
        for row, text in enumerate(lines):
            color = (0, 0, 255) if text.startswith("EVENT") else (255, 255, 255)
            y = int((22 + row * 22) * scale)
            cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale,
                        (0, 0, 0), max(3, int(4 * scale)))
            cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale,
                        color, max(1, int(scale)))

        with self.view_lock:
            self.latest[stream_name] = view

    def frames(self) -> List[Tuple[str, np.ndarray]]:
        with self.view_lock:
            return list(self.latest.items())


def _fit(view: np.ndarray, min_width: int = 960, max_width: int = 1280) -> np.ndarray:
    height, width = view.shape[:2]
    target = min(max(width, min_width), max_width)
    if target == width:
        return view
    ratio = target / width
    return cv2.resize(view, (target, int(height * ratio)))


async def main(config_path: str) -> None:
    system = LiveViewSystem(config_path)
    run_task = asyncio.create_task(system.run())
    last_tick = {}

    try:
        while not run_task.done():
            for stream_name, view in system.frames():
                if last_tick.get(stream_name) is view:
                    continue
                last_tick[stream_name] = view
                cv2.imshow(f"Highway Incident Detection - {stream_name}", _fit(view))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            await asyncio.sleep(0.01)
    finally:
        system.running = False
        if system._shutdown_event:
            system._shutdown_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/local_gpu.yaml")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.config))
    except KeyboardInterrupt:
        pass
