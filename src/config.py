from pathlib import Path


RAW_DATA_PATH = Path("data/raw/METABRIC_RNA_Mutation.csv")
CLEAN_DATA_PATH = Path("data/cleaned/metabric_clean.csv")
TARGET_COLUMN = "pam50_+_claudin-low_subtype"

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
