from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
SPLITS_DIR = DATA_ROOT / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, CHECKPOINTS_DIR, PLOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset configuration
MCD_RPPG_DATASET_ID = "kyegorov/mcd_rppg"  # HuggingFace ID
TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# Video processing
FRAME_RATE = 30
WINDOW_SIZE = 30  # frames (1 second @ 30fps)
WINDOW_OVERLAP = 0.5  # 50% overlap
FACE_ROI_SIZE = 128  # pixels (128x128)

# Model architecture
FEATURE_DIM = 256
HIDDEN_DIM = 128
NUM_TASKS = 3  # HR, RR, SpO2

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-5

# Loss weights
LOSS_WEIGHTS = {
    "hr": 1.0,
    "rr": 1.0,
    "spo2": 1.0,
    "confidence": 0.5
}

# Target ranges for normalization
TARGET_RANGES = {
    "hr": (40, 200),           # bpm
    "rr": (8, 60),             # breaths/min
    "spo2": (80, 100)          # percentage
}

# Device
DEVICE = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
