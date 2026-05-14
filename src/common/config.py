import os
from pathlib import Path

# Thư mục gốc của dự án
ROOT = Path(__file__).resolve().parent.parent.parent

# Các thư mục chính
SRC_DIR = ROOT / "src"
DATA_DIR = SRC_DIR / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

# Thư mục con trong Artifacts
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
SPLITS_DIR = ARTIFACTS_DIR / "splits"

# Đảm bảo các thư mục tồn tại
for directory in [ARTIFACTS_DIR, PREDICTIONS_DIR, REPORTS_DIR, SPLITS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Đường dẫn tới các dataset
TYPE1_PATH = DATA_DIR / "Logic_Based_Educational_Queries_Text_Only" / "Logic_Based_Educational_Queries.json"
TYPE2_PATH = DATA_DIR / "Physics_Problems_Text_Only.csv"
NORMALIZED_DATA_PATH = ARTIFACTS_DIR / "normalized_dataset.jsonl"

# Cấu hình chia split (mặc định)
SPLIT_RATIOS = {
    "train": 0.70,
    "dev": 0.15,
    "test": 0.15,
}
DEFAULT_SEED = 42

# Tùy chọn Model
MODEL_ID = os.getenv("EXACT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MOCK_LLM = os.getenv("MOCK_LLM", "0") == "1"
MAX_NEW_TOKENS = int(os.getenv("EXACT_MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("EXACT_TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("EXACT_TOP_P", "1.0"))
