"""
Streamlit Cloud Entry Point for Sonaris
Delegates execution to app/app.py
"""
import sys
import runpy
from pathlib import Path

# Resolve project root
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

APP_FILE = ROOT_DIR / "app" / "app.py"

if __name__ == "__main__":
    runpy.run_path(str(APP_FILE), run_name="__main__")
