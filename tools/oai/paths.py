"""
Filesystem layout helpers.

Every path the toolkit reads or writes is resolved through here so the rest of
the code never hardcodes a directory. FRAMEWORK_ROOT is discovered by walking up
from this file until we find the marker files that identify the repo root.
"""

from __future__ import annotations

from pathlib import Path

_MARKERS = ("config", "vllm", "catalog")


def framework_root() -> Path:
    """Return the repo root (the folder that holds config/, vllm/, catalog/)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if all((parent / m).exists() for m in _MARKERS):
            return parent
    # Fallback: tools/oai/paths.py -> repo root is two levels up.
    return here.parents[2]


ROOT = framework_root()

CONFIG_ENV = ROOT / "config" / "config.env"
CATALOG_DIR = ROOT / "catalog" / "models"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

VLLM_MODELS_DIR = ROOT / "vllm" / "models"
MODEL_DOWNLOAD_DIR = ROOT / "model-download"
MANIFESTS_DIR = ROOT / "configs" / "manifests"
PROFILES_DIR = ROOT / "configs" / "workload_profiles"
RESULTS_DIR = ROOT / "results"


def catalog_path(model_id: str) -> Path:
    return CATALOG_DIR / f"{model_id}.yaml"


def model_dir(model_id: str) -> Path:
    return VLLM_MODELS_DIR / model_id


def download_dir(model_id: str) -> Path:
    return MODEL_DOWNLOAD_DIR / model_id
