"""
generate all figures and visualizations for the project.

produces:
  - scaling curves (f1/accuracy vs model size, per classifier)
  - confusion matrices (per model × best classifier)
  - one-vs-rest roc curves
  - pca / umap / t-sne embedding space plots
  - per-class mcc bar charts
  - statistical comparison tables

usage:
    python scripts/visualize.py
    python scripts/visualize.py --model esm2_150m --skip-embeddings
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from utils.plot_utils import (
    plot_scaling_curve,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_embedding_space,
    plot_per_class_mcc,
)
from utils.eval_utils import wilcoxon_compare

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_all_results() -> dict:
    """load results.json from all model directories."""
    all_results = {}
    for model_name in cfg.models:
        path = cfg.results_dir / model_name / "results.json"
        if path.exists():
            with open(path) as f:
                all_results[model_name] = json.load(f)
            logger.info(f"loaded results for {model_name}")
    return all_results


def generate_scaling_curves(results: dict, fig_dir: Path):
    """generate f1 and accuracy scaling curves."""
    for metric in ["macro_f1", "weighted_f1", "accuracy", "avg_mcc"]:
        path = fig_dir / f"scaling_{metric}.png"
        plot_scaling_curve(results, metric=metric, save_path=path)
        plt.close()


def generate_confusion_matrices(results: dict, fig_dir: Path):
    """generate confusion matrices for each model's best classifier."""
    for model_name, clf_results in results.items():
        # find best classifier by macro f1
        best_clf = max(clf_results, key=lambda k: clf_results[k].get("macro_f1", 0))
        cm = np.array(clf_results[best_clf]["confusion_matrix"])

        path = fig_dir / f"cm_{model_name}_{best_clf}.png"
        title = f"Confusion Matrix — {model_name} + {best_clf.replace('_', ' ').title()}"
        plot_confusion_matrix(cm, title=title, save_path=path)
        plt.close()

        logger.info(f"  best classifier for {model_name}: {best_clf}")


def generate_embedding_plots(fig_dir: Path, methods=("pca", "tsne")):
    """generate 2d embedding visualizations for each model."""
    from utils.data_utils import encode_labels

    for model_name in cfg.models:
        emb_path = cfg.emb_dir / model_name / "test_embeddings.npy"
        meta_path = cfg.emb_dir / model_name / "test_metadata.npz"

        if not emb_path.exists():
            logger.warning(f"no test embeddings for {model_name}, skipping")
            continue

        x = np.load(emb_path)
        meta = np.load(meta_path, allow_pickle=True)
        y, _ = encode_labels(meta["labels"].tolist())

        for method in methods:
            path = fig_dir / f"embed_{model_name}_{method}.png"
            title = f"{model_name.upper()} — {method.upper()}"

            try:
                plot_embedding_space(
                    x, y, method=method, title=title, save_path=path
                )
                plt.close()
            except ImportError as e:
                logger.warning(f"skipping {method} for {model_name}: {e}")


def generate_per_class_mcc(results: dict, fig_dir: Path):
    """generate per-class mcc bar charts."""
    for model_name, clf_results in results.items():
        path = fig_dir / f"mcc_{model_name}.png"
        plot_per_class_mcc(clf_results, model_name, save_path=path)
        plt.close()


def generate_stat_comparisons(results: dict, fig_dir: Path):
    """
    run wilcoxon tests between adjacent model sizes for each classifier.
    saves results as a text table.
    """
    model_order = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"]
    available = [m for m in model_order if m in results]

    if len(available) < 2:
        logger.info("need at least 2 models for statistical comparison")
        return

    lines = ["Statistical Comparisons (Wilcoxon Signed-Rank Test)", "=" * 60, ""]

    # get all classifiers present in any model
    all_clfs = set()
    for m in available:
        all_clfs.update(results[m].keys())

    for clf_name in sorted(all_clfs):
        lines.append(f"\nClassifier: {clf_name}")
        lines.append("-" * 40)

        for i in range(len(available) - 1):
            m_a, m_b = available[i], available[i + 1]
            r_a = results.get(m_a, {}).get(clf_name, {})
            r_b = results.get(m_b, {}).get(clf_name, {})

            cv_a = r_a.get("cv_scores", [])
            cv_b = r_b.get("cv_scores", [])

            if len(cv_a) != len(cv_b) or len(cv_a) < 3:
                lines.append(f"  {m_a} vs {m_b}: insufficient data")
                continue

            try:
                result = wilcoxon_compare(np.array(cv_a), np.array(cv_b))
                sig = "*" if result["significant"] else " "
                lines.append(
                    f"  {m_a} vs {m_b}: "
                    f"p={result['p_value']:.4f} {sig} "
                    f"(stat={result['statistic']:.2f})"
                )
            except Exception as e:
                lines.append(f"  {m_a} vs {m_b}: error - {e}")

    out_path = fig_dir / "statistical_comparisons.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"saved statistical comparisons to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="generate project visualizations")
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="skip embedding space visualizations (slow)",
    )
    parser.add_argument(
        "--umap", action="store_true",
        help="include umap plots (requires umap-learn)",
    )
    args = parser.parse_args()

    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # load all results
    results = load_all_results()
    if not results:
        logger.error("no results found. run train_classifiers.py first.")
        return

    # generate all visualizations
    logger.info("\n── scaling curves ──")
    generate_scaling_curves(results, fig_dir)

    logger.info("\n── confusion matrices ──")
    generate_confusion_matrices(results, fig_dir)

    logger.info("\n── per-class mcc ──")
    generate_per_class_mcc(results, fig_dir)

    if not args.skip_embeddings:
        logger.info("\n── embedding space plots ──")
        methods = ["pca", "tsne"]
        if args.umap:
            methods.append("umap")
        generate_embedding_plots(fig_dir, methods=methods)

    logger.info("\n── statistical comparisons ──")
    generate_stat_comparisons(results, fig_dir)

    logger.info(f"\nall figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
