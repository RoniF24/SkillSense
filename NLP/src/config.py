from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
SYNTH_DIR = DATA_DIR / "synthetic"
PROC_DIR = DATA_DIR / "processed"
