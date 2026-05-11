from __future__ import annotations

import argparse
from pathlib import Path


def export_engine(
    model_path: Path,
    imgsz: int,
    device: str,
    half: bool,
    int8: bool,
    workspace: int | None,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    kwargs: dict = {
        "format": "engine",
        "imgsz": imgsz,
        "device": device,
        "half": half,
        "int8": int8,
    }
    if workspace is not None:
        kwargs["workspace"] = workspace
    exported = model.export(**kwargs)
    return Path(exported)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO .pt model to TensorRT .engine for Jetson/Edge deployment.")
    parser.add_argument("--model", type=Path, required=True, help="Input .pt model path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0 on Jetson/PC GPU.")
    parser.add_argument("--half", action="store_true", help="Export FP16 engine.")
    parser.add_argument("--int8", action="store_true", help="Export INT8 engine. Requires calibration support/data in Ultralytics setup.")
    parser.add_argument("--workspace", type=int, default=None, help="TensorRT workspace size in GiB if supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    output = export_engine(
        model_path=args.model,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        int8=args.int8,
        workspace=args.workspace,
    )
    print(f"Exported TensorRT engine: {output}")


if __name__ == "__main__":
    main()