from pathlib import Path


# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_MAIN_DIR = RESULTS_DIR / "main"
RESULTS_ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
RESULTS_PER_CLASS_DIR = RESULTS_DIR / "per_class"
RESULTS_PREDICTIONS_DIR = RESULTS_DIR / "predictions"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"


MODELS_DIR = PROJECT_ROOT / "models"
TRANSFORMER_RUNS_DIR = MODELS_DIR / "transformer"

DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_project_directories():
    """Create output directories if they do not exist."""

    directories = [
        RAW_DATA_DIR,
        REFERENCE_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_MAIN_DIR,
        RESULTS_ROBUSTNESS_DIR,
        RESULTS_PER_CLASS_DIR,
        RESULTS_PREDICTIONS_DIR,
        RESULTS_TABLES_DIR,
        TRANSFORMER_RUNS_DIR,
        DOCS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# DATA FILES

FSTEC_RAW_FILE = (
    RAW_DATA_DIR
    / "vullist.xlsx"
)

RUSSIAN_VENDORS_FILE = (
    REFERENCE_DATA_DIR
    / "russian_vendors.csv"
)


# DataPrep outputs

RU_MODELLING_FILE = (
    INTERIM_DATA_DIR
    / "FSTEC_Russian_30CWE_Modelling_Dataset.xlsx"
)

NONRU_MODELLING_FILE = (
    INTERIM_DATA_DIR
    / "FSTEC_NonRussian_30CWE_Modelling_Dataset.xlsx"
)


# LeakageGroups outputs

RU_LEAKAGE_FILE = (
    INTERIM_DATA_DIR
    / "FSTEC_Russian_30CWE_with_leakage_groups.xlsx"
)

NONRU_LEAKAGE_FILE = (
    INTERIM_DATA_DIR
    / "FSTEC_NonRussian_30CWE_with_leakage_groups.xlsx"
)


# DataPartitioning outputs

RU_PARTITIONS_FILE = (
    INTERIM_DATA_DIR
    / "FSTEC_Russian_30CWE_Fixed_Partitions.xlsx"
)

NONRU_PARTITIONS_FILE = (
    INTERIM_DATA_DIR
    / "FSTEC_NonRussian_30CWE_Fixed_Partitions.xlsx"
)


# TextPreprocessing outputs

RU_PREPROCESSED_FILE = (
    PROCESSED_DATA_DIR
    / "FSTEC_Russian_30CWE_Preprocessed.xlsx"
)

NONRU_PREPROCESSED_FILE = (
    PROCESSED_DATA_DIR
    / "FSTEC_NonRussian_30CWE_Preprocessed.xlsx"
)


# DATASET COLUMNS

RAW_TEXT_COL = "Описание уязвимости"
TEXT_COL = "text_clean"

LABEL_COL = "Тип ошибки CWE"

GROUP_COL = "leakage_group_id"
ID_COL = "ml_row_id"

SPLIT_COL = "split"
DOMAIN_COL = "domain"

BDU_COL = "Идентификатор"

SOFTWARE_COL = "Тип ПО"
DATE_COL = "Дата публикации"
YEAR_COL = "year"


# REPRODUCIBILITY

RANDOM_STATE = 42
SEED = 42


# FIXED CWE LABEL SPACE

CWE_LABELS = [
    "CWE-119",
    "CWE-120",
    "CWE-121",
    "CWE-122",
    "CWE-125",
    "CWE-190",
    "CWE-20",
    "CWE-200",
    "CWE-22",
    "CWE-264",
    "CWE-269",
    "CWE-284",
    "CWE-287",
    "CWE-362",
    "CWE-399",
    "CWE-400",
    "CWE-401",
    "CWE-404",
    "CWE-415",
    "CWE-416",
    "CWE-476",
    "CWE-665",
    "CWE-770",
    "CWE-78",
    "CWE-787",
    "CWE-79",
    "CWE-835",
    "CWE-843",
    "CWE-89",
    "CWE-94",
]

N_CLASSES = len(CWE_LABELS)


# TEXT PREPROCESSING

URL_TOKEN = "urltoken"
CWE_TOKEN = "cwetoken"


# CLASSICAL ML SETTINGS

WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)

TFIDF_MIN_DF = 2

WORD_TFIDF_MAX_FEATURES = 100_000
CHAR_TFIDF_MAX_FEATURES = 200_000

LOGISTIC_REGRESSION_C = 1.0
SVM_C = 1.0


# EVALUATION

PRIMARY_METRIC = "macro_f1"

SECONDARY_METRICS = [
    "weighted_f1",
    "accuracy",
]


# RuModernBERT

MODEL_ID = "deepvk/RuModernBERT-base"

MAX_LENGTH = 256

LEARNING_RATE = 2e-5

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16

GRADIENT_ACCUMULATION_STEPS = 2

WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0.10