"""Portable project paths and constants, no dependencies on current working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
STATIC_DIR = ROOT / "static"
SEED = 20261027
HORIZON_STEPS = 3  # 3 x 5 minutes: predict 15 minutes into the future
STEP_MINUTES = 5
GATES = [f"G{i}" for i in range(1, 7)]
