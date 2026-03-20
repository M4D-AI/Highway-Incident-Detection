from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "pipeline_benchmark.py"
)
_SPEC = spec_from_file_location("pipeline_benchmark", _BENCHMARK_PATH)
_MODULE = module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

BenchmarkResult = _MODULE.BenchmarkResult
build_scenarios = _MODULE.build_scenarios
parse_csv_ints = _MODULE.parse_csv_ints
parse_csv_strings = _MODULE.parse_csv_strings
render_markdown_table = _MODULE.render_markdown_table


def test_benchmark_helpers_expand_scenarios():
    args = Namespace(
        video="data/parking.mp4",
        duration_seconds=5.0,
        frame_skips="1,2",
        max_pending_frames="1,3",
        processing_ms=10.0,
        handler_profiles="none,mixed",
        handler_delay_ms=5.0,
        emit_every_n_frames=20,
    )

    scenarios = build_scenarios(args)

    assert parse_csv_ints("1,2,4") == [1, 2, 4]
    assert parse_csv_strings("none,mixed") == ["none", "mixed"]
    assert len(scenarios) == 8
    assert scenarios[0].video_path == "data/parking.mp4"


def test_benchmark_markdown_table_contains_metrics():
    table = render_markdown_table(
        [
            BenchmarkResult(
                scenario="skip=1|pending=2|handler=none",
                elapsed_seconds=5.0,
                accepted_frames=100,
                processed_frames=90,
                emitted_results=3,
                dropped_frames=10,
                pending_frames=0,
                effective_fps=18.0,
                avg_callback_ms=12.5,
                p95_callback_ms=18.0,
                avg_emit_ms=4.0,
                p95_emit_ms=5.5,
            )
        ]
    )

    assert "skip=1|pending=2|handler=none" in table
    assert "| fps |" in table
    assert "18.00" in table
