import gc
import random

import numpy as np
import torch

from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from src.config import (
    SEED,

    TEXT_COL,
    LABEL_COL,

    CWE_LABELS,
    N_CLASSES,

    MODEL_ID,
    MAX_LENGTH,

    LEARNING_RATE,

    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,

    GRADIENT_ACCUMULATION_STEPS,

    WEIGHT_DECAY,
    WARMUP_STEPS,
)


# LABEL MAPPING

LABEL2ID = {
    label: index
    for index, label
    in enumerate(CWE_LABELS)
}

ID2LABEL = {
    index: label
    for label, index
    in LABEL2ID.items()
}


# REPRODUCIBILITY

def set_global_seed(
    seed=SEED
):
    """Set random seeds for reproducible transformer runs."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


# HARDWARE

def get_precision_settings():
    """
    Return BF16/FP16 settings for the current hardware.
    """

    use_bf16 = (
        torch.cuda.is_available()
        and
        torch.cuda.is_bf16_supported()
    )

    use_fp16 = (
        torch.cuda.is_available()
        and
        not use_bf16
    )

    return use_bf16, use_fp16


def print_device_info():
    """Print PyTorch and GPU information."""

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        memory_gb = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            /
            1024 ** 3
        )

        print(
            "GPU memory:",
            round(memory_gb, 2),
            "GB"
        )

    use_bf16, use_fp16 = (
        get_precision_settings()
    )

    print(
        "BF16:",
        use_bf16
    )

    print(
        "FP16:",
        use_fp16
    )


# TOKENIZER

def make_tokenizer():
    """Load the fixed RuModernBERT tokenizer."""

    return AutoTokenizer.from_pretrained(
        MODEL_ID
    )


# DATASET

def make_hf_dataset(
    df,
    tokenizer
):
    """
    Convert a project dataframe into a tokenised
    Hugging Face Dataset.
    """

    temp = (
        df[
            [
                TEXT_COL,
                LABEL_COL
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    temp["labels"] = (
        temp[LABEL_COL]
        .map(LABEL2ID)
    )

    if temp["labels"].isna().any():

        unknown = (
            temp.loc[
                temp["labels"].isna(),
                LABEL_COL
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown CWE labels: {unknown}"
        )

    temp["labels"] = (
        temp["labels"]
        .astype(int)
    )

    dataset = Dataset.from_pandas(
        temp[
            [
                TEXT_COL,
                "labels"
            ]
        ],
        preserve_index=False
    )

    def tokenize_batch(batch):

        return tokenizer(
            batch[TEXT_COL],

            truncation=True,

            max_length=MAX_LENGTH
        )

    return dataset.map(
        tokenize_batch,

        batched=True,

        remove_columns=[
            TEXT_COL
        ]
    )


# DATA COLLATOR

def make_data_collator(
    tokenizer
):
    """Create dynamic-padding collator."""

    return DataCollatorWithPadding(
        tokenizer=tokenizer
    )


# CLASS WEIGHTS

def calculate_class_weights(
    train_df
):
    """
    Calculate balanced inverse-frequency weights
    for the current training dataset.
    """

    counts = (
        train_df[LABEL_COL]
        .value_counts()
        .reindex(CWE_LABELS)
    )

    if counts.isna().any():

        missing = (
            counts[
                counts.isna()
            ]
            .index
            .tolist()
        )

        raise ValueError(
            f"Training data is missing CWE classes: {missing}"
        )

    n_samples = len(train_df)
    n_classes = len(CWE_LABELS)

    weights = (
        n_samples
        /
        (
            n_classes
            *
            counts.values
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# WEIGHTED LOSS

def make_weighted_loss(
    class_weights
):
    """
    Return a weighted cross-entropy function
    compatible with Hugging Face Trainer.
    """

    def weighted_loss(
        outputs,
        labels,
        num_items_in_batch=None
    ):

        logits = outputs.logits

        weights = class_weights.to(
            logits.device
        )

        return (
            torch.nn.functional
            .cross_entropy(
                logits,
                labels,
                weight=weights
            )
        )

    return weighted_loss


# MODEL

def make_model():
    """Load a fresh RuModernBERT classification model."""

    return (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_ID,

            num_labels=N_CLASSES,

            id2label=ID2LABEL,

            label2id=LABEL2ID
        )
    )


# TRAINER METRICS

def compute_metrics(
    eval_pred
):
    """Metrics used during Hugging Face evaluation."""

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    return {
        "macro_f1":
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            ),

        "weighted_f1":
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),
    }


# TRAINING ARGUMENTS

def make_training_arguments(
    output_dir,
    num_epochs,
    evaluation=False
):
    """
    Create common RuModernBERT TrainingArguments.

    The notebook decides whether the run is
    validation/model-selection or final training.
    """

    use_bf16, use_fp16 = (
        get_precision_settings()
    )

    arguments = {
        "output_dir":
            str(output_dir),

        "num_train_epochs":
            num_epochs,

        "learning_rate":
            LEARNING_RATE,

        "per_device_train_batch_size":
            TRAIN_BATCH_SIZE,

        "per_device_eval_batch_size":
            EVAL_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "weight_decay":
            WEIGHT_DECAY,

        "warmup_steps":
            WARMUP_STEPS,

        "lr_scheduler_type":
            "linear",

        # Logging
        "logging_strategy":
            "epoch",

        "disable_tqdm":
            False,

        "log_level":
            "warning",

        # Precision
        "bf16":
            use_bf16,

        "fp16":
            use_fp16,

        # External logging disabled
        "report_to":
            "none",

        # Reproducibility
        "seed":
            SEED,

        "data_seed":
            SEED,

        # Safer on Windows
        "dataloader_num_workers":
            0,
    }

    if evaluation:

        arguments.update({
            "eval_strategy":
                "epoch",

            "save_strategy":
                "epoch",

            "load_best_model_at_end":
                True,

            "metric_for_best_model":
                "macro_f1",

            "greater_is_better":
                True,

            "save_total_limit":
                1,
        })

    else:

        arguments.update({
            "eval_strategy":
                "no",

            "save_strategy":
                "no",
        })

    return TrainingArguments(
        **arguments
    )


# CLEANUP

def cleanup_cuda():
    """Release Python and CUDA memory after a transformer run."""

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()