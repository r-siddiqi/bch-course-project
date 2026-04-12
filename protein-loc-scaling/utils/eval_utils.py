"""
evaluation utilities — metrics computation, bootstrap ci, statistical tests.
"""

import logging
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import config as cfg

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[list[str]] = None,
) -> dict:
    """
    compute full classification metrics suite.

    args:
        y_true: true integer labels.
        y_pred: predicted integer labels.
        y_prob: predicted probabilities (n_samples, n_classes).
        labels: class names for reporting.

    returns:
        dict with accuracy, macro_f1, weighted_f1, per_class_mcc,
        confusion_matrix, and optionally roc_auc.
    """
    labels = labels or cfg.labels
    n_cls = len(labels)

    # basic metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # per-class mcc (one-vs-rest)
    per_class_mcc = {}
    for i, name in enumerate(labels):
        binary_true = (y_true == i).astype(int)
        binary_pred = (y_pred == i).astype(int)
        if binary_true.sum() == 0:
            per_class_mcc[name] = float("nan")
        else:
            per_class_mcc[name] = matthews_corrcoef(binary_true, binary_pred)

    # overall mcc (for multiclass, average the per-class values)
    valid_mccs = [v for v in per_class_mcc.values() if not np.isnan(v)]
    avg_mcc = np.mean(valid_mccs) if valid_mccs else 0.0

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))

    result = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "avg_mcc": avg_mcc,
        "per_class_mcc": per_class_mcc,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true, y_pred, target_names=labels, zero_division=0
        ),
    }

    # roc auc (one-vs-rest) — requires probability estimates
    if y_prob is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(n_cls)))
            # macro auc
            roc_auc_macro = roc_auc_score(
                y_bin, y_prob, average="macro", multi_class="ovr"
            )
            # per-class auc
            per_class_auc = {}
            for i, name in enumerate(labels):
                if y_bin[:, i].sum() > 0:
                    per_class_auc[name] = roc_auc_score(y_bin[:, i], y_prob[:, i])
                else:
                    per_class_auc[name] = float("nan")

            result["roc_auc_macro"] = roc_auc_macro
            result["per_class_auc"] = per_class_auc
        except ValueError as e:
            logger.warning(f"could not compute roc auc: {e}")

    return result


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable = f1_score,
    n_boot: int = cfg.n_bootstrap,
    ci: float = cfg.ci_level,
    seed: int = cfg.random_state,
    **metric_kwargs,
) -> tuple[float, float, float]:
    """
    compute bootstrap confidence interval for a metric.

    returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            s = metric_fn(y_true[idx], y_pred[idx], **metric_kwargs)
            scores.append(s)
        except ValueError:
            continue

    scores = np.array(scores)
    point = metric_fn(y_true, y_pred, **metric_kwargs)
    alpha = (1 - ci) / 2
    lo = np.percentile(scores, 100 * alpha)
    hi = np.percentile(scores, 100 * (1 - alpha))

    return point, lo, hi


def wilcoxon_compare(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict:
    """
    wilcoxon signed-rank test between two sets of per-fold scores.
    tests whether model a and model b have significantly different performance.

    args:
        scores_a: array of scores from model a (e.g., per fold or per bootstrap).
        scores_b: array of scores from model b.

    returns:
        dict with statistic, p_value, and significant (at 0.05).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("score arrays must have equal length")

    diff = np.array(scores_a) - np.array(scores_b)

    # if all differences are zero, no significant difference
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < 0.05,
    }
