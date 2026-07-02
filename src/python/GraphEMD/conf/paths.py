"""Canonical paths for the GraphEMD repository."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PYTHON = REPO_ROOT / "src" / "python"
DATA_DIR = REPO_ROOT / "data"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DOCS_DIR = REPO_ROOT / "docs"
PAPER_FIGURES_DIR = REPO_ROOT.parent / "PAPER" / "figures"
