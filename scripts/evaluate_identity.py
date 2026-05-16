from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from edge_inspector.identity.encoder import HistogramIdentityEncoder, normalize_embedding

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_bgr(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return rgb[:, :, ::-1].copy()


def class_images(root: Path) -> dict[str, list[Path]]:
    classes: dict[str, list[Path]] = {}
    if not root.exists():
        return classes
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            classes[class_dir.name] = images
    return classes


def build_prototypes(
    encoder: HistogramIdentityEncoder,
    support: dict[str, list[Path]],
    shots: int,
) -> tuple[list[str], np.ndarray]:
    class_names: list[str] = []
    prototypes: list[np.ndarray] = []
    for class_name, images in sorted(support.items()):
        selected = images[:shots]
        if len(selected) < shots:
            continue
        embeddings = [encoder.encode(read_bgr(path)) for path in selected]
        prototype = normalize_embedding(np.mean(np.stack(embeddings), axis=0))
        class_names.append(class_name)
        prototypes.append(prototype)
    if not prototypes:
        return [], np.empty((0, 0), dtype=np.float32)
    return class_names, np.stack(prototypes).astype(np.float32)


def predict(encoder: HistogramIdentityEncoder, image_path: Path, class_names: list[str], prototypes: np.ndarray) -> tuple[str | None, float]:
    if not class_names:
        return None, 0.0
    embedding = encoder.encode(read_bgr(image_path))
    similarities = prototypes @ embedding
    idx = int(np.argmax(similarities))
    return class_names[idx], float(similarities[idx])


def evaluate_known(
    encoder: HistogramIdentityEncoder,
    support: dict[str, list[Path]],
    query: dict[str, list[Path]],
    shots: int,
) -> dict:
    class_names, prototypes = build_prototypes(encoder, support, shots)
    total = 0
    correct = 0
    similarities: list[float] = []
    for true_class, images in sorted(query.items()):
        if true_class not in class_names:
            continue
        for path in images:
            pred_class, score = predict(encoder, path, class_names, prototypes)
            total += 1
            correct += int(pred_class == true_class)
            similarities.append(score)
    return {
        "shots": shots,
        "classes": len(class_names),
        "queries": total,
        "accuracy": float(correct / total) if total else 0.0,
        "avg_similarity": float(np.mean(similarities)) if similarities else 0.0,
    }


def evaluate_unknown(
    encoder: HistogramIdentityEncoder,
    seen_support: dict[str, list[Path]],
    unseen_query: dict[str, list[Path]],
    shots: int,
    threshold: float,
) -> dict:
    class_names, prototypes = build_prototypes(encoder, seen_support, shots)
    total = 0
    rejected = 0
    similarities: list[float] = []
    for images in unseen_query.values():
        for path in images:
            _, score = predict(encoder, path, class_names, prototypes)
            total += 1
            rejected += int(score < threshold)
            similarities.append(score)
    return {
        "shots": shots,
        "queries": total,
        "unknown_threshold": threshold,
        "unknown_recall": float(rejected / total) if total else 0.0,
        "avg_best_similarity": float(np.mean(similarities)) if similarities else 0.0,
    }


def evaluate_fewshot_enroll(
    encoder: HistogramIdentityEncoder,
    unseen_support_query: dict[str, list[Path]],
    shots: int,
) -> dict:
    support: dict[str, list[Path]] = {}
    query: dict[str, list[Path]] = {}
    for class_name, images in unseen_support_query.items():
        if len(images) <= shots:
            continue
        support[class_name] = images[:shots]
        query[class_name] = images[shots:]
    return evaluate_known(encoder, support, query, shots)


def run(args: argparse.Namespace) -> dict:
    encoder = HistogramIdentityEncoder(image_size=args.image_size, hist_bins=args.hist_bins)
    seen_train = class_images(args.seen_root / "train")
    seen_valid = class_images(args.seen_root / "valid")
    unseen_train = class_images(args.unseen_root / "train")

    report = {
        "seen_root": str(args.seen_root),
        "unseen_root": str(args.unseen_root),
        "encoder": encoder.name,
        "threshold": args.threshold,
        "known": [],
        "unknown": [],
        "fewshot_enroll": [],
    }
    for shots in args.shots:
        report["known"].append(evaluate_known(encoder, seen_train, seen_valid, shots))
        report["unknown"].append(evaluate_unknown(encoder, seen_train, unseen_train, shots, args.threshold))
        report["fewshot_enroll"].append(evaluate_fewshot_enroll(encoder, unseen_train, shots))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate open-set label identity on seen/unseen folder datasets.")
    parser.add_argument("--seen-root", type=Path, default=Path("datasets/dataLabelClassification_seen"))
    parser.add_argument("--unseen-root", type=Path, default=Path("datasets/dataLabelClassification_unseen"))
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hist-bins", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=Path("reports/identity_eval.json"))
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
