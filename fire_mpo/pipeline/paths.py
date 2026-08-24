"""Path helpers for preference_dataset/{model}/{dataset}/greedy/..."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFERENCE_ROOT = PROJECT_ROOT / "preference_dataset"
REJECTED_IMAGES_ROOT = PROJECT_ROOT / "medsam3_rejected_images"
MODELS_ROOT = PROJECT_ROOT / "models"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

SUPPORTED_DATASETS = ("slake", "vqa_rad", "iu_xray")


def preference_dir(model: str, dataset: str, split: str = "greedy") -> Path:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")
    return PREFERENCE_ROOT / model / dataset / split


def preference_json(
    model: str,
    dataset: str,
    name: str = "rrpo_with_medsam3_rejected.json",
    split: str = "greedy",
) -> Path:
    return preference_dir(model, dataset, split) / name


def rejected_images_dir(dataset: str) -> Path:
    return REJECTED_IMAGES_ROOT / dataset


def resolve_optional(path: Optional[str], default: Path) -> Path:
    return Path(path) if path else default
