"""Central path configuration for the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "fruit_dataset"
DATA_YAML = DATA_DIR / "data.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
YOLO_WEIGHTS = MODELS_DIR / "yolo_finetuned" / "weights" / "best.pt"
VIT_MODEL_DIR = MODELS_DIR / "vit_finetuned"
SAM_WEIGHTS = MODELS_DIR / "sam_finetuned" / "sam_decoder_best.pth"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
CURVES_DIR = OUTPUTS_DIR / "curves"

METRICS_YOLO = METRICS_DIR / "metrics_yolo.json"
METRICS_VIT = METRICS_DIR / "metrics_vit.json"
METRICS_SAM = METRICS_DIR / "metrics_sam.json"

CURVE_YOLO = CURVES_DIR / "training_curves_yolo.png"
CURVE_VIT = CURVES_DIR / "training_curves_vit.png"
CURVE_SAM = CURVES_DIR / "training_curves_sam.png"

YOLO_RUN_DIR = PROJECT_ROOT / "runs" / "detect" / "models" / "yolo_finetuned-3"
YOLO_RUN_WEIGHTS = YOLO_RUN_DIR / "weights" / "best.pt"
YOLO_RESULTS_CSV = YOLO_RUN_DIR / "results.csv"
YOLO_RESULTS_PNG = YOLO_RUN_DIR / "results.png"

PRETRAINED_YOLO = PROJECT_ROOT / "yolov8n.pt"
