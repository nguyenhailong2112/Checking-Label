from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from edge_inspector.core.inspector import LabelBarcodeInspector
from edge_inspector.utils.config import load_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_bgr_image(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return rgb[:, :, ::-1].copy()


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    index = min(len(arr) - 1, max(0, round((pct / 100.0) * (len(arr) - 1))))
    return float(arr[index])


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps_avg": 0.0}
    avg_ms = mean(latencies_ms)
    return {
        "count": len(latencies_ms),
        "avg_ms": float(avg_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "fps_avg": float(1000.0 / avg_ms) if avg_ms > 0 else 0.0,
    }


def run_benchmark(config_path: Path, input_path: Path, warmup: int, limit: int | None, output_json: Path | None) -> dict:
    cfg = load_config(config_path)
    inspector = LabelBarcodeInspector(cfg)
    image_paths = collect_images(input_path)
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_path}")

    warmup_image = read_bgr_image(image_paths[0])
    for _ in range(max(0, warmup)):
        inspector.run(warmup_image, image_name=image_paths[0].name)

    latencies_ms: list[float] = []
    decisions = {"OK": 0, "NG": 0}
    for path in image_paths:
        image = read_bgr_image(path)
        start = time.perf_counter()
        result, _, _ = inspector.run(image, image_name=path.name)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        decisions[result.decision] += 1
        print(f"{path.name}: {result.decision} | {elapsed_ms:.1f} ms | conf={result.total_confidence:.3f}")

    report = {
        "config_path": str(config_path),
        "input_path": str(input_path),
        "summary": summarize_latencies(latencies_ms),
        "decisions": decisions,
        "latencies_ms": latencies_ms,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Edge AI inspection pipeline on an image folder.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Runtime YAML config path.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder to benchmark.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs on the first image.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_benchmark(args.config, args.input, args.warmup, args.limit, args.output_json)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()