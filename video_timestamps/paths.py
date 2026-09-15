"""Locations shared by the web app and command-line tools."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_default_home = PROJECT_ROOT if (PROJECT_ROOT / "pyproject.toml").exists() else Path.cwd()
DATA_HOME = Path(os.environ.get("VIDEO_TIMESTAMPS_HOME", _default_home)).expanduser().resolve()
OUTPUT_DIR = DATA_HOME / "outputs"
INPUT_DIR = DATA_HOME / "data"
