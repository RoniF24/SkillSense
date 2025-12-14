from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "Prompts"

UNIFIED_PROMPT = (PROMPTS_DIR / "unified_prompt.txt").read_text(encoding="utf-8")
