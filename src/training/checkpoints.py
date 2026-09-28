"""Checkpoints JSON portáveis, sem serializar datasets ou pesos no Git."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_checkpoint(path: Path, epoch: int, model_state: dict[str, float], optimizer_state: dict[str, float], metrics: dict[str, float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"epoch": epoch, "model": model_state, "optimizer": optimizer_state, "metrics": metrics}, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not {"epoch", "model", "optimizer", "metrics"}.issubset(data):
        raise ValueError("Checkpoint inválido.")
    return data
