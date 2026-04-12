"""
visualization utilities — scaling curves, confusion matrices, roc plots,
embedding space visualizations (pca, umap, t-sne).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import config as cfg

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

logger = logging.getLogger(__name__)


# ── color palettes ─────────────────────────────────────────────────
# distinct colors for classifiers and model sizes
clf_colors = {
    "logistic_regression": "#1f77b4",
    "random_forest":       "#2ca02c",
    "svm":                 "#d62728",
    "knn":                 "#9467bd",
    "xgboost":             "#ff7f0e",
    "mlp":                 "#8c564b",
}

model_colors = {
    "esm2_8m":   "#a6cee3",
    "esm2_35m":  "#1f78b4",
    "esm2_150m": "#33a02c",
    "esm2_650m": "#e31a1c",
}

# tab10 for 10 localization classes
loc_cmap = plt.cm.tab10


def plot_scaling_curve(
    results: dict,
    metric: str = "macro_f1",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    plot accuracy/f1 vs model size, one line per classifier.

    args:
        results: nested dict {model_name: {clf_name: {metric: value, ...}}}.
        metric: which metric to plot on y-axis.
        save_path: if provided, saves figure to this path.

    returns:
        matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # extract model sizes for x-axis (in millions of parameters)
    model_sizes = {
        "esm2_8m": 8,
        "esm2_35m": 35,
        "esm2_150m": 150,
        "esm2_650m": 650,
    }

    # collect all classifier names across all models
    all_clfs = set()
    for model in results.values():
        all_clfs.update(model.keys())

    for clf_name in sorted(all_clfs):
        xs, ys, lo, hi = [], [], [], []
        for model_name in ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"]:
            if model_name not in results:
                continue
            if clf_name not in results[model_name]:
                continue
            r = results[model_name][clf_name]
            xs.append(model_sizes[model_name])
            ys.append(r[metric])
            # add ci if available
            ci_key = f"{metric}_ci"
            if ci_key in r:
                lo.append(r[ci_key][0])
                hi.append(r[ci_key][1])

        color = clf_colors.get(clf_name, "#333333")
        ax.plot(xs, ys, "o-", label=clf_name.replace("_", " ").title(),
                color=color, linewidth=2, markersize=6)

        if lo and hi:
            ax.fill_between(xs, lo, hi, alpha=0.15, color=color)

    ax.set_xscale("log")
    ax.set_xticks([8, 35, 150, 650])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("ESM-2 Model Size (millions of parameters)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Subcellular Localization: {metric.replace('_', ' ').title()} vs Model Size")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        logger.info(f"saved scaling curve to {save_path}")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    plot a normalized confusion matrix heatmap.
    """
    labels = labels or cfg.labels

    # normalize by row (true label)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    if save_path:
        fig.savefig(save_path)
        logger.info(f"saved confusion matrix to {save_path}")

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: Optional[list[str]] = None,
    title: str = "One-vs-Rest ROC Curves",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    plot one-vs-rest roc curves for all classes.
    """
    labels = labels or cfg.labels
    n_cls = len(labels)

    y_bin = label_binarize(y_true, classes=list(range(n_cls)))

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, name in enumerate(labels):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})",
                color=loc_cmap(i / n_cls), linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        logger.info(f"saved roc curves to {save_path}")

    return fig


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    class_names: Optional[list[str]] = None,
    title: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    2d visualization of embedding space using pca, umap, or t-sne.

    args:
        embeddings: (n_samples, emb_dim) array.
        labels: integer class labels.
        method: one of 'pca', 'umap', 'tsne'.
        class_names: string names for legend.
        title: plot title.
        save_path: output path.
    """
    class_names = class_names or cfg.labels

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=cfg.random_state)
        coords = reducer.fit_transform(embeddings)
        ax_labels = (
            f"PC1 ({reducer.explained_variance_ratio_[0]*100:.1f}%)",
            f"PC2 ({reducer.explained_variance_ratio_[1]*100:.1f}%)",
        )
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=cfg.random_state,
                       perplexity=30, n_iter=1000)
        coords = reducer.fit_transform(embeddings)
        ax_labels = ("t-SNE 1", "t-SNE 2")
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=cfg.random_state,
                            n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(embeddings)
        ax_labels = ("UMAP 1", "UMAP 2")
    else:
        raise ValueError(f"unknown method: {method}. use pca, tsne, or umap.")

    fig, ax = plt.subplots(figsize=(9, 7))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=8, alpha=0.5, label=name,
                   color=loc_cmap(lbl / len(class_names)))

    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    ax.set_title(title or f"{method.upper()} Embedding Visualization")
    ax.legend(loc="best", fontsize=7, markerscale=3, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    if save_path:
        fig.savefig(save_path)
        logger.info(f"saved {method} plot to {save_path}")

    return fig


def plot_per_class_mcc(
    results: dict,
    model_name: str,
    labels: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    grouped bar chart of per-class mcc across classifiers for one model.

    args:
        results: {clf_name: {per_class_mcc: {class: val, ...}}}.
        model_name: name for the title.
        labels: class names.
    """
    labels = labels or cfg.labels
    clf_names = sorted(results.keys())
    n_cls = len(labels)
    n_clf = len(clf_names)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_cls)
    width = 0.8 / n_clf

    for i, clf in enumerate(clf_names):
        mcc_vals = [
            results[clf].get("per_class_mcc", {}).get(lbl, 0.0)
            for lbl in labels
        ]
        color = clf_colors.get(clf, "#333333")
        offset = (i - n_clf / 2 + 0.5) * width
        ax.bar(x + offset, mcc_vals, width, label=clf.replace("_", " ").title(),
               color=color, alpha=0.85)

    ax.set_xlabel("Subcellular Compartment")
    ax.set_ylabel("Matthews Correlation Coefficient")
    ax.set_title(f"Per-Class MCC — {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    if save_path:
        fig.savefig(save_path)
        logger.info(f"saved per-class mcc to {save_path}")

    return fig
