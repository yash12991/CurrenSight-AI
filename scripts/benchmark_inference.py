"""Benchmark CurrenSight AI denomination inference on one local image."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend import main  # noqa: E402


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def benchmark(image_path, iterations=30, warmups=3):
    image_path = Path(image_path).resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not decode benchmark image: {image_path}")
    if main.PYTORCH_MODEL is None and main.MODEL is None:
        raise RuntimeError("No denomination model loaded")

    for _ in range(warmups):
        main.perform_inference(image)

    timings = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter()
        result = main.perform_inference(image)
        timings.append((time.perf_counter() - started) * 1000)

    mean_ms = statistics.mean(timings)
    return {
        "image": str(image_path),
        "engine": "pytorch" if main.PYTORCH_MODEL is not None else "classic_cv",
        "device": str(main.PYTORCH_DEVICE) if main.PYTORCH_MODEL is not None else "cpu",
        "iterations": iterations,
        "warmups": warmups,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings),
        "p95_ms": percentile(timings, 0.95),
        "throughput_fps": 1000 / mean_ms,
        "last_result": result,
    }


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        default=PROJECT_ROOT / "data" / "images" / "val" / "val_500_rupees_0.jpg",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0:
        parser.error("iterations must be positive and warmups cannot be negative")
    print(json.dumps(benchmark(args.image, args.iterations, args.warmups), indent=2))


if __name__ == "__main__":
    main_cli()
