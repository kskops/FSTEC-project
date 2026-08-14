import numpy as np
import time

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.svm import (
    LinearSVC
)

from src.config import (
    RANDOM_STATE,

    WORD_NGRAM_RANGE,
    CHAR_NGRAM_RANGE,

    TFIDF_MIN_DF,

    WORD_TFIDF_MAX_FEATURES,
    CHAR_TFIDF_MAX_FEATURES,

    LOGISTIC_REGRESSION_C,
    SVM_C,
)


# TF-IDF

def make_word_tfidf():
    """Create the fixed word-level TF-IDF representation."""

    return TfidfVectorizer(
        analyzer="word",

        ngram_range=WORD_NGRAM_RANGE,

        token_pattern=(
            r"(?u)\b[\w][\w.+#/-]*"
        ),

        lowercase=False,

        min_df=TFIDF_MIN_DF,

        max_features=WORD_TFIDF_MAX_FEATURES,

        sublinear_tf=True,

        norm="l2",

        dtype=np.float32,
    )


def make_char_tfidf():
    """Create the fixed character-level TF-IDF representation."""

    return TfidfVectorizer(
        analyzer="char_wb",

        ngram_range=CHAR_NGRAM_RANGE,

        lowercase=False,

        min_df=TFIDF_MIN_DF,

        max_features=CHAR_TFIDF_MAX_FEATURES,

        sublinear_tf=True,

        norm="l2",

        dtype=np.float32,
    )


# CLASSIFIERS

def make_logistic_regression():
    """Create the Logistic Regression classifier."""

    return LogisticRegression(
        C=LOGISTIC_REGRESSION_C,

        solver="lbfgs",

        max_iter=2000,

        class_weight="balanced",
    )


def make_linear_svm():
    """Create the Linear SVM classifier."""

    return LinearSVC(
        C=SVM_C,

        class_weight="balanced",

        dual=True,

        max_iter=5000,

        random_state=RANDOM_STATE,
    )

def fit_predict_classical(
    train_df,
    test_df,
    model_name,
    representation,
    text_col,
    label_col
):
    """
    Fit one classical text classifier and return predictions.

    The vectorizer is fitted only on the supplied training data.
    """

    if representation == "word":
        vectorizer = make_word_tfidf()

    elif representation == "char":
        vectorizer = make_char_tfidf()

    else:
        raise ValueError(
            "representation must be 'word' or 'char'"
        )


    if model_name == "Logistic Regression":
        model = make_logistic_regression()

    elif model_name == "Linear SVM":
        model = make_linear_svm()

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )


    start = time.time()

    X_train = vectorizer.fit_transform(
        train_df[text_col]
    )

    X_test = vectorizer.transform(
        test_df[text_col]
    )

    model.fit(
        X_train,
        train_df[label_col]
    )

    predictions = model.predict(
        X_test
    )

    runtime = time.time() - start

    return (
        predictions,
        X_train.shape[1],
        runtime
    )