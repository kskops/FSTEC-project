from pathlib import Path

import pandas as pd

from src.config import (
    FSTEC_RAW_FILE,
    RUSSIAN_VENDORS_FILE,

    RU_MODELLING_FILE,
    NONRU_MODELLING_FILE,

    RU_LEAKAGE_FILE,
    NONRU_LEAKAGE_FILE,

    RU_PARTITIONS_FILE,
    NONRU_PARTITIONS_FILE,

    RU_PREPROCESSED_FILE,
    NONRU_PREPROCESSED_FILE,

    LABEL_COL,
    GROUP_COL,
    ID_COL,
    SPLIT_COL,
    TEXT_COL,

    CWE_LABELS,
)


# GENERIC HELPERS

def require_file(path):
    """Return path if it exists, otherwise raise an error."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    return path


def validate_columns(
    df,
    required_columns,
    dataset_name="dataset"
):
    """Check that all required columns are present."""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )


def validate_label_space(
    df,
    expected_labels=CWE_LABELS,
    label_col=LABEL_COL,
    dataset_name="dataset"
):
    """Check whether a dataframe contains the expected labels."""

    observed = set(
        df[label_col]
        .dropna()
        .astype(str)
        .unique()
    )

    expected = set(expected_labels)

    missing = expected - observed
    unexpected = observed - expected

    if missing or unexpected:
        raise ValueError(
            f"{dataset_name} label-space mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Unexpected: {sorted(unexpected)}"
        )


def select_split(
    df,
    split_name,
    split_col=SPLIT_COL
):
    """Return a copy containing one named partition."""

    return (
        df.loc[
            df[split_col].eq(split_name)
        ]
        .copy()
        .reset_index(drop=True)
    )


def assert_no_group_overlap(
    first_df,
    second_df,
    group_col=GROUP_COL
):
    """Check that two datasets do not share leakage groups."""

    overlap = (
        set(first_df[group_col])
        &
        set(second_df[group_col])
    )

    if overlap:
        raise ValueError(
            f"Detected {len(overlap)} overlapping leakage groups."
        )

    return True


# RAW DATA

def load_raw_fstec(
    path=FSTEC_RAW_FILE
):
    """Load the original FSTEC BDU workbook."""

    path = require_file(path)

    return pd.read_excel(
        path,
        sheet_name="Уязвимости",
        header=2
    )


def load_russian_vendors(
    path=RUSSIAN_VENDORS_FILE
):
    """Load the curated Russian-vendor lookup."""

    path = require_file(path)

    return pd.read_csv(
        path,
        encoding="utf-8-sig"
    )


# DATAPREP OUTPUTS

def load_modelling_data():
    """Load the RU and Non-RU 30-CWE modelling datasets."""

    ru = pd.read_excel(
        require_file(RU_MODELLING_FILE)
    )

    nonru = pd.read_excel(
        require_file(NONRU_MODELLING_FILE)
    )

    validate_columns(
        ru,
        [LABEL_COL],
        "Russian modelling dataset"
    )

    validate_columns(
        nonru,
        [LABEL_COL],
        "Non-Russian modelling dataset"
    )

    return ru, nonru


# LEAKAGE-GROUP OUTPUTS

def load_leakage_data():
    """Load datasets containing leakage_group_id."""

    ru = pd.read_excel(
        require_file(RU_LEAKAGE_FILE)
    )

    nonru = pd.read_excel(
        require_file(NONRU_LEAKAGE_FILE)
    )

    required = [
        ID_COL,
        LABEL_COL,
        GROUP_COL,
    ]

    validate_columns(
        ru,
        required,
        "Russian leakage dataset"
    )

    validate_columns(
        nonru,
        required,
        "Non-Russian leakage dataset"
    )

    return ru, nonru


# FIXED PARTITIONS

def load_fixed_partition_data():
    """Load the fixed group-aware partition datasets."""

    ru = pd.read_excel(
        require_file(RU_PARTITIONS_FILE)
    )

    nonru = pd.read_excel(
        require_file(NONRU_PARTITIONS_FILE)
    )

    required = [
        ID_COL,
        LABEL_COL,
        GROUP_COL,
        SPLIT_COL,
    ]

    validate_columns(
        ru,
        required,
        "Russian fixed partitions"
    )

    validate_columns(
        nonru,
        required,
        "Non-Russian fixed partitions"
    )

    return ru, nonru


# PREPROCESSED DATA

def load_preprocessed_data():
    """Load the final text-preprocessed datasets."""

    ru = pd.read_excel(
        require_file(RU_PREPROCESSED_FILE)
    )

    nonru = pd.read_excel(
        require_file(NONRU_PREPROCESSED_FILE)
    )

    required = [
        ID_COL,
        LABEL_COL,
        GROUP_COL,
        SPLIT_COL,
        TEXT_COL,
    ]

    validate_columns(
        ru,
        required,
        "Russian preprocessed dataset"
    )

    validate_columns(
        nonru,
        required,
        "Non-Russian preprocessed dataset"
    )

    return ru, nonru