from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "cv_output"
DEFAULT_MODEL_NAME = "yolov8n.pt"
