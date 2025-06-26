# src/config.py

from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "airbnb.csv"
