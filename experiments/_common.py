"""Lets every script be run as `python experiments/<name>.py` from the repository root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
