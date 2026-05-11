from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from edge_inspector.utils.config import AppConfig
from edge_inspector.utils.time import utc_now


@dataclass(frozen=True)
class ModelArtifactRecord:
    target: str
    source_name: str
    artifact_path: Path
    timestamp: str
    note: str = ""

    def to_json(self) -> dict[str, str]:
        return {
            "target": self.target,
            "source_name": self.source_name,
            "artifact_path": str(self.artifact_path),
            "timestamp": self.timestamp,
            "note": self.note,
        }


class ModelRegistry:
    """Stage model artifacts and keep a small JSONL manifest for quick rollback/audit."""

    VALID_TARGETS = {"label", "code", "defect"}
    VALID_SUFFIXES = {".pt", ".engine"}

    def __init__(self, config: AppConfig) -> None:
        self.deploy_dir = Path(config.get("models.deploy_dir", "weights/staged"))
        self.manifest_path = self.deploy_dir / "manifest.jsonl"

    def stage_artifact(self, target: str, source_name: str, payload: bytes, note: str = "") -> ModelArtifactRecord:
        target = target.lower()
        if target not in self.VALID_TARGETS:
            raise ValueError(f"Unsupported model target: {target}")

        suffix = Path(source_name).suffix.lower()
        if suffix not in self.VALID_SUFFIXES:
            raise ValueError(f"Unsupported model artifact suffix: {suffix}")

        self.deploy_dir.mkdir(parents=True, exist_ok=True)
        timestamp = utc_now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = Path(source_name).name.replace(" ", "_")
        destination = self.deploy_dir / f"{target}_{timestamp}_{safe_name}"
        destination.write_bytes(payload)

        record = ModelArtifactRecord(
            target=target,
            source_name=Path(source_name).name,
            artifact_path=destination,
            timestamp=timestamp,
            note=note,
        )
        self._append_manifest(record)
        return record

    def import_existing(self, target: str, source_path: Path, note: str = "") -> ModelArtifactRecord:
        payload = source_path.read_bytes()
        return self.stage_artifact(target=target, source_name=source_path.name, payload=payload, note=note)

    def list_artifacts(self, target: str | None = None) -> list[ModelArtifactRecord]:
        if not self.manifest_path.exists():
            return []

        records: list[ModelArtifactRecord] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            record = ModelArtifactRecord(
                target=item["target"],
                source_name=item["source_name"],
                artifact_path=Path(item["artifact_path"]),
                timestamp=item["timestamp"],
                note=item.get("note", ""),
            )
            if target is None or record.target == target:
                records.append(record)
        return records

    def remove_missing_from_manifest(self) -> None:
        records = [record for record in self.list_artifacts() if record.artifact_path.exists()]
        self.deploy_dir.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    def copy_to_weights(self, record: ModelArtifactRecord, weights_dir: Path = Path("weights")) -> Path:
        """Optional production helper: copy a staged artifact to a stable weights path."""

        weights_dir.mkdir(parents=True, exist_ok=True)
        destination = weights_dir / f"{record.target}_model{record.artifact_path.suffix}"
        shutil.copy2(record.artifact_path, destination)
        return destination

    def _append_manifest(self, record: ModelArtifactRecord) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")