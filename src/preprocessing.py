import re
import unicodedata

import pandas as pd

from src.config import (
    RAW_TEXT_COL,
    TEXT_COL,
    LABEL_COL,
    URL_TOKEN,
    CWE_TOKEN,
)


# REGEX PATTERNS

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE
)

CWE_PATTERN = re.compile(
    r"\bCWE-\d+\b",
    flags=re.IGNORECASE
)


# TEXT QUALITY

def text_quality_summary(
    df,
    text_col=RAW_TEXT_COL
):
    """Return basic missing/empty-description statistics."""

    text = (
        df[text_col]
        .fillna("")
        .astype(str)
    )

    return {
        "records": len(df),

        "missing_descriptions":
            int(df[text_col].isna().sum()),

        "empty_descriptions":
            int(
                text
                .str.strip()
                .eq("")
                .sum()
            ),
    }


# CWE LEAKAGE AUDIT

def extract_cwe_mentions(text):
    """Extract explicit CWE identifiers from text."""

    if pd.isna(text):
        return []

    return sorted(
        {
            value.upper()
            for value
            in CWE_PATTERN.findall(str(text))
        }
    )


def cwe_leakage_summary(
    df,
    text_col=RAW_TEXT_COL,
    label_col=LABEL_COL
):
    """
    Count descriptions containing explicit CWE identifiers
    and descriptions containing their own target CWE.
    """

    mentions = (
        df[text_col]
        .apply(extract_cwe_mentions)
    )

    contains_any = (
        mentions
        .apply(bool)
    )

    contains_target = pd.Series(
        [
            str(label).upper() in row_mentions

            for label, row_mentions
            in zip(
                df[label_col],
                mentions
            )
        ],
        index=df.index
    )

    return {
        "descriptions_with_any_cwe":
            int(contains_any.sum()),

        "descriptions_with_target_cwe":
            int(contains_target.sum()),
    }


# TEXT PREPROCESSING

def preprocess_text(text):
    """
    Apply the fixed dissertation text-normalisation protocol.

    Applied:
        NFKC Unicode normalisation
        lowercase
        ё -> е
        dash normalisation
        quotation normalisation
        URL replacement
        explicit CWE-ID replacement
        whitespace normalisation
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalisation
    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # Lowercase
    text = text.lower()

    # Russian ё -> е
    text = text.replace(
        "ё",
        "е"
    )

    # Normalise different dash characters
    text = re.sub(
        r"[–—−]",
        "-",
        text
    )

    # Normalise quotation marks
    text = (
        text
        .replace("«", '"')
        .replace("»", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
    )

    # Replace URLs
    text = URL_PATTERN.sub(
        f" {URL_TOKEN} ",
        text
    )

    # Remove direct CWE-label leakage
    text = CWE_PATTERN.sub(
        f" {CWE_TOKEN} ",
        text
    )

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def preprocess_dataframe(
    df,
    input_col=RAW_TEXT_COL,
    output_col=TEXT_COL
):
    """Return a dataframe copy with a cleaned text column."""

    result = df.copy()

    result[output_col] = (
        result[input_col]
        .apply(preprocess_text)
    )

    return result


# VALIDATION

def validate_preprocessed_text(
    df,
    text_col=TEXT_COL
):
    """Check cleaned descriptions for obvious problems."""

    cleaned = (
        df[text_col]
        .fillna("")
        .astype(str)
    )

    explicit_cwe_remaining = int(
        cleaned
        .str.contains(
            CWE_PATTERN,
            regex=True
        )
        .sum()
    )

    empty_after_preprocessing = int(
        cleaned
        .str.strip()
        .eq("")
        .sum()
    )

    return {
        "records":
            len(df),

        "empty_after_preprocessing":
            empty_after_preprocessing,

        "explicit_cwe_ids_remaining":
            explicit_cwe_remaining,
    }