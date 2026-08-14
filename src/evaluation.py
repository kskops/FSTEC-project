import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import (
    CWE_LABELS
)


# MAIN EVALUATION

def evaluate_predictions(
    y_true,
    y_pred,
    labels=CWE_LABELS
):
    """
    Calculate aggregate and per-class classification metrics.
    """

    labels = list(labels)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    (
        precision,
        recall,
        f1,
        support
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    aggregate = {
        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "accuracy":
            accuracy,
    }

    per_class = pd.DataFrame({
        "CWE":
            labels,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "support":
            support,
    })

    return aggregate, per_class


# RQ METRICS

def domain_generalisation_gap(
    ru_to_ru_macro_f1,
    nonru_to_ru_macro_f1
):
    """
    Difference between in-domain and cross-domain Macro-F1.
    """

    return (
        ru_to_ru_macro_f1
        -
        nonru_to_ru_macro_f1
    )


def combined_training_gain(
    ru_to_ru_macro_f1,
    combined_to_ru_macro_f1
):
    """
    Difference between Combined->RU and RU->RU Macro-F1.
    """

    return (
        combined_to_ru_macro_f1
        -
        ru_to_ru_macro_f1
    )


# CONFUSION MATRIX

def confusion_matrix_dataframe(
    y_true,
    y_pred,
    labels=CWE_LABELS
):
    """Return a labelled confusion matrix."""

    labels = list(labels)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    return pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )


# PER-CWE COMPARISON

def compare_per_class_f1(
    first_df,
    second_df,
    first_name,
    second_name
):
    """Compare F1 scores for two conditions CWE by CWE."""

    first = (
        first_df[
            ["CWE", "f1"]
        ]
        .rename(
            columns={
                "f1":
                    f"{first_name}_f1"
            }
        )
    )

    second = (
        second_df[
            ["CWE", "f1"]
        ]
        .rename(
            columns={
                "f1":
                    f"{second_name}_f1"
            }
        )
    )

    comparison = first.merge(
        second,
        on="CWE",
        how="outer"
    )

    comparison[
        "f1_difference"
    ] = (
        comparison[
            f"{first_name}_f1"
        ]
        -
        comparison[
            f"{second_name}_f1"
        ]
    )

    return comparison