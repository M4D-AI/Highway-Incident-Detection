"""Pipeline throughput benchmark for stream scheduling and result dispatch."""

import argparse
import asyncio
import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from highway_incident_detection.config.settings import OutputConfig, StreamConfig
from highway_incident_detection.detectors.event_types import EventType, IncidentEvent
from highway_incident_detection.output.result_output import EventResult, ResultHandler
from highway_incident_detection.stream.rtsp_stream import RTSPStreamProcessor
from highway_incident_detection.summary.summary_generator import SummaryGenerator


@dataclass
class BenchmarkScenario:
    """One benchmark configuration."""

    video_path: str
    duration_seconds: float
    frame_skip: int
    max_pending_frames: int
    processing_ms: float
    handler_profile: str
    handler_delay_ms: float
    emit_every_n_frames: int

    @property
    def name(self) -> str:
        return (
            f"skip={self.frame_skip}|pending={self.max_pending_frames}|"
            f"handler={self.handler_profile}"
        )


@dataclass
class BenchmarkResult:
    """Measured benchmark output."""

    scenario: str
    elapsed_seconds: float
    accepted_frames: int
    processed_frames: int
    emitted_results: int
    dropped_frames: int
    pending_frames: int
    effective_fps: float
    avg_callback_ms: float
    p95_callback_ms: float
    avg_emit_ms: float
    p95_emit_ms: float


class SyncSleepHandler:
    """Synchronous handler that simulates downstream work."""

    def __init__(self, delay_ms: float):
        self.delay_seconds = max(0.0, delay_ms / 1000.0)

    def __call__(self, result: EventResult) -> None:
        time.sleep(self.delay_seconds)


class AsyncSleepHandler:
    """Asynchronous handler that simulates downstream work."""

    def __init__(self, delay_ms: float):
        self.delay_seconds = max(0.0, delay_ms / 1000.0)

    async def __call__(self, result: EventResult) -> None:
        await asyncio.sleep(self.delay_seconds)


def parse_csv_ints(value: str) -> List[int]:
    """Parse comma-separated integers."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> List[str]:
    """Parse comma-separated strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def percentile(values: Sequence[float], ratio: float) -> float:
    """Return a percentile value from a numeric sequence."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * ratio) - 1))
    return float(sorted_values[index])


def build_handlers(profile: str, delay_ms: float) -> List[ResultHandler]:
    """Build benchmark handlers for one profile."""
    if profile == "none":
        return []
    if profile == "sync":
        return [SyncSleepHandler(delay_ms)]
    if profile == "async":
        return [AsyncSleepHandler(delay_ms)]
    if profile == "mixed":
        return [SyncSleepHandler(delay_ms), AsyncSleepHandler(delay_ms)]
    raise ValueError(
        f"Unsupported handler profile: {profile}. "
        "Expected one of: none, sync, async, mixed"
    )


def build_synthetic_result(frame_number: int, stream_name: str) -> EventResult:
    """Build a synthetic event result for output-layer benchmarks."""
    event = IncidentEvent(
        event_type=EventType.CONGESTION,
        stream_name=stream_name,
        timestamp=datetime.now(),
        frame_number=frame_number,
        confidence=0.9,
        description="Synthetic benchmark event",
        metadata={"benchmark": True},
    )
    return EventResult(
        event=event,
        event_id=f"{stream_name}-frame-{frame_number}",
        keyframe_path="",
        video_path="",
        summary_path="",
    )


async def run_scenario(scenario: BenchmarkScenario) -> BenchmarkResult:
    """Run one benchmark scenario."""
    callback_durations: List[float] = []
    emit_durations: List[float] = []
    processed_frames = 0
    emitted_results = 0

    with tempfile.TemporaryDirectory(prefix="hid-benchmark-") as temp_dir:
        output_dir = Path(temp_dir)
        summary_generator = SummaryGenerator(
            OutputConfig(
                output_dir=str(output_dir),
                keyframe_dir=str(output_dir / "keyframes"),
                video_dir=str(output_dir / "videos"),
                summary_dir=str(output_dir / "summaries"),
                audit_dir=str(output_dir / "audits"),
            ),
            result_handlers=build_handlers(
                scenario.handler_profile,
                scenario.handler_delay_ms,
            ),
        )

        async def frame_callback(stream_name, frame, frame_number, timestamp):
            nonlocal processed_frames, emitted_results

            callback_start = time.perf_counter()
            await asyncio.sleep(max(0.0, scenario.processing_ms / 1000.0))
            processed_frames += 1

            if (
                scenario.emit_every_n_frames > 0
                and frame_number % scenario.emit_every_n_frames == 0
            ):
                emit_start = time.perf_counter()
                await summary_generator._emit_result(
                    build_synthetic_result(frame_number, stream_name)
                )
                emit_durations.append((time.perf_counter() - emit_start) * 1000.0)
                emitted_results += 1

            callback_durations.append((time.perf_counter() - callback_start) * 1000.0)

        processor = RTSPStreamProcessor(
            StreamConfig(
                streams=[
                    {
                        "name": "benchmark_stream",
                        "url": scenario.video_path,
                        "enabled": True,
                    }
                ],
                frame_skip=scenario.frame_skip,
                max_pending_frames=scenario.max_pending_frames,
                drop_frames_when_busy=True,
                auto_reconnect=True,
                reconnect_interval=1,
                heartbeat_interval=30,
                no_frame_timeout=300,
            ),
            frame_callback=frame_callback,
        )

        start_time = time.perf_counter()
        stream_task = asyncio.create_task(processor.start())
        sleep_task = asyncio.create_task(asyncio.sleep(scenario.duration_seconds))
        done, _ = await asyncio.wait(
            [stream_task, sleep_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if sleep_task in done and not stream_task.done():
            await processor.stop()

        await asyncio.gather(stream_task, sleep_task, return_exceptions=True)
        elapsed = time.perf_counter() - start_time
        status = processor.get_stream_status().get("benchmark_stream", {})

    effective_fps = processed_frames / elapsed if elapsed > 0 else 0.0
    return BenchmarkResult(
        scenario=scenario.name,
        elapsed_seconds=elapsed,
        accepted_frames=int(status.get("frame_count", 0)),
        processed_frames=processed_frames,
        emitted_results=emitted_results,
        dropped_frames=int(status.get("dropped_frames", 0)),
        pending_frames=int(status.get("pending_frames", 0)),
        effective_fps=effective_fps,
        avg_callback_ms=(sum(callback_durations) / len(callback_durations))
        if callback_durations
        else 0.0,
        p95_callback_ms=percentile(callback_durations, 0.95),
        avg_emit_ms=(sum(emit_durations) / len(emit_durations)) if emit_durations else 0.0,
        p95_emit_ms=percentile(emit_durations, 0.95),
    )


def build_scenarios(args: argparse.Namespace) -> List[BenchmarkScenario]:
    """Expand CLI args into a scenario matrix."""
    frame_skips = parse_csv_ints(args.frame_skips)
    max_pending_frames = parse_csv_ints(args.max_pending_frames)
    handler_profiles = parse_csv_strings(args.handler_profiles)

    scenarios = []
    for frame_skip in frame_skips:
        for pending in max_pending_frames:
            for profile in handler_profiles:
                scenarios.append(
                    BenchmarkScenario(
                        video_path=args.video,
                        duration_seconds=args.duration_seconds,
                        frame_skip=frame_skip,
                        max_pending_frames=pending,
                        processing_ms=args.processing_ms,
                        handler_profile=profile,
                        handler_delay_ms=args.handler_delay_ms,
                        emit_every_n_frames=args.emit_every_n_frames,
                    )
                )
    return scenarios


def render_markdown_table(results: Sequence[BenchmarkResult]) -> str:
    """Render results as a markdown table."""
    lines = [
        "| scenario | elapsed_s | accepted | processed | dropped | emitted | fps | avg_cb_ms | p95_cb_ms | avg_emit_ms | p95_emit_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {scenario} | {elapsed:.2f} | {accepted} | {processed} | {dropped} | "
            "{emitted} | {fps:.2f} | {avg_cb:.2f} | {p95_cb:.2f} | {avg_emit:.2f} | {p95_emit:.2f} |".format(
                scenario=result.scenario,
                elapsed=result.elapsed_seconds,
                accepted=result.accepted_frames,
                processed=result.processed_frames,
                dropped=result.dropped_frames,
                emitted=result.emitted_results,
                fps=result.effective_fps,
                avg_cb=result.avg_callback_ms,
                p95_cb=result.p95_callback_ms,
                avg_emit=result.avg_emit_ms,
                p95_emit=result.p95_emit_ms,
            )
        )
    return "\n".join(lines)


async def async_main(args: argparse.Namespace) -> List[BenchmarkResult]:
    """Run the full benchmark suite."""
    results = []
    for scenario in build_scenarios(args):
        print(f"[benchmark] running {scenario.name}")
        results.append(await run_scenario(scenario))
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark stream scheduling and result-dispatch throughput",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/parking.mp4",
        help="Path to the local video used as the stream source",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=10.0,
        help="How long each scenario should run",
    )
    parser.add_argument(
        "--frame-skips",
        type=str,
        default="1,2,4",
        help="Comma-separated frame_skip values",
    )
    parser.add_argument(
        "--max-pending-frames",
        type=str,
        default="1,2,4",
        help="Comma-separated max_pending_frames values",
    )
    parser.add_argument(
        "--processing-ms",
        type=float,
        default=20.0,
        help="Simulated per-frame processing time in milliseconds",
    )
    parser.add_argument(
        "--handler-profiles",
        type=str,
        default="none,sync,async,mixed",
        help="Comma-separated handler profiles: none,sync,async,mixed",
    )
    parser.add_argument(
        "--handler-delay-ms",
        type=float,
        default=10.0,
        help="Simulated per-handler delay in milliseconds",
    )
    parser.add_argument(
        "--emit-every-n-frames",
        type=int,
        default=30,
        help="Emit one synthetic result every N processed frames; 0 disables result emission",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path for saving JSON benchmark output",
    )
    args = parser.parse_args()

    results = asyncio.run(async_main(args))
    print()
    print(render_markdown_table(results))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([asdict(item) for item in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved JSON benchmark report to {output_path}")


if __name__ == "__main__":
    main()
