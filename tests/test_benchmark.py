import importlib.util
from pathlib import Path


def load_benchmark_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_latencies_returns_fps_and_percentiles() -> None:
    benchmark = load_benchmark_module()
    summary = benchmark.summarize_latencies([10.0, 20.0, 30.0])

    assert summary["count"] == 3
    assert summary["avg_ms"] == 20.0
    assert summary["fps_avg"] == 50.0
    assert summary["p50_ms"] == 20.0


def test_summarize_latencies_handles_empty_input() -> None:
    benchmark = load_benchmark_module()
    summary = benchmark.summarize_latencies([])

    assert summary["count"] == 0
    assert summary["fps_avg"] == 0.0