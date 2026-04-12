"""
train and evaluate classifiers on esm-2 embeddings.

trains six classifiers (logistic regression, random forest, svm, knn,
xgboost, mlp) on mean-pooled embeddings from each esm-2 model variant.
uses stratified 5-fold cross-validation for hyperparameter tuning and
evaluates on the held-out test set.

usage:
    python scripts/train_classifiers.py --model esm2_8m
    python scripts/train_classifiers.py --model all
    python scripts/train_classifiers.py --model esm2_150m --clf svm xgboost
"""

import sys
import json
import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from utils.data_utils import encode_labels
from utils.eval_utils import compute_metrics, bootstrap_ci

# suppress convergence warnings during grid search
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_embeddings(model_name: str, split: str) -> tuple:
    """
    load pre-extracted embeddings and labels from disk.

    returns:
        (embeddings, labels_str, accessions)
    """
    emb_dir = cfg.emb_dir / model_name
    emb_path = emb_dir / f"{split}_embeddings.npy"
    meta_path = emb_dir / f"{split}_metadata.npz"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"embeddings not found at {emb_path}. "
            f"run scripts/extract_embeddings.py --model {model_name} first."
        )

    x = np.load(emb_path)
    meta = np.load(meta_path, allow_pickle=True)

    return x, meta["labels"], meta["accessions"]


def build_classifier(clf_name: str) -> tuple:
    """
    build a sklearn pipeline with scaler + classifier and param grid.

    returns:
        (pipeline, param_grid) for use with gridsearchcv.
    """
    params = cfg.clf_params[clf_name]

    if clf_name == "logistic_regression":
        clf = LogisticRegression(
            max_iter=params["max_iter"],
            class_weight=params["class_weight"],
            solver=params["solver"],
            multi_class=params["multi_class"],
            random_state=cfg.random_state,
        )
        grid = {"clf__C": params["C"]}

    elif clf_name == "random_forest":
        clf = RandomForestClassifier(
            class_weight=params["class_weight"],
            n_jobs=params["n_jobs"],
            random_state=cfg.random_state,
        )
        grid = {
            "clf__n_estimators": params["n_estimators"],
            "clf__max_depth": params["max_depth"],
        }

    elif clf_name == "svm":
        clf = SVC(
            kernel=params["kernel"],
            class_weight=params["class_weight"],
            decision_function_shape=params["decision_function_shape"],
            probability=True,  # needed for roc curves
            random_state=cfg.random_state,
        )
        grid = {"clf__C": params["C"]}

    elif clf_name == "knn":
        clf = KNeighborsClassifier(
            weights=params["weights"],
            metric=params["metric"],
        )
        grid = {"clf__n_neighbors": params["n_neighbors"]}

    elif clf_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("xgboost not installed. pip install xgboost")
            return None, None

        clf = XGBClassifier(
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            eval_metric=params["eval_metric"],
            use_label_encoder=params["use_label_encoder"],
            random_state=cfg.random_state,
            verbosity=0,
        )
        grid = {
            "clf__n_estimators": params["n_estimators"],
            "clf__max_depth": params["max_depth"],
            "clf__learning_rate": params["learning_rate"],
        }

    elif clf_name == "mlp":
        clf = MLPClassifier(
            activation=params["activation"],
            early_stopping=params["early_stopping"],
            validation_fraction=params["validation_fraction"],
            max_iter=params["max_iter"],
            random_state=cfg.random_state,
        )
        grid = {
            "clf__hidden_layer_sizes": params["hidden_layer_sizes"],
            "clf__learning_rate_init": params["learning_rate_init"],
        }

    else:
        raise ValueError(f"unknown classifier: {clf_name}")

    # wrap in pipeline with standard scaler
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    return pipe, grid


def train_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    clf_name: str,
) -> dict:
    """
    train classifier with grid search cv, evaluate on test set.

    args:
        x_train: training embeddings (n_train, emb_dim).
        y_train: training labels (n_train,).
        x_test: test embeddings (n_test, emb_dim).
        y_test: test labels (n_test,).
        clf_name: classifier name from cfg.clf_params.

    returns:
        dict with metrics, best params, and predictions.
    """
    logger.info(f"\n  training {clf_name}...")
    t0 = time.time()

    pipe, grid = build_classifier(clf_name)
    if pipe is None:
        return {}

    # stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    search = GridSearchCV(
        pipe, grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)

    # best model predictions on test set
    y_pred = search.predict(x_test)

    # get probability estimates if available
    y_prob = None
    if hasattr(search.best_estimator_, "predict_proba"):
        y_prob = search.predict_proba(x_test)

    # compute full metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # bootstrap confidence intervals for macro f1
    f1_point, f1_lo, f1_hi = bootstrap_ci(
        y_test, y_pred,
        average="macro", zero_division=0,
    )
    metrics["macro_f1_ci"] = (f1_lo, f1_hi)

    # store cv scores for statistical comparison
    metrics["cv_scores"] = search.cv_results_["mean_test_score"].tolist()
    metrics["best_params"] = {
        k.replace("clf__", ""): v
        for k, v in search.best_params_.items()
    }
    metrics["best_cv_score"] = float(search.best_score_)

    elapsed = time.time() - t0
    logger.info(
        f"  {clf_name}: macro_f1={metrics['macro_f1']:.4f} "
        f"[{f1_lo:.4f}, {f1_hi:.4f}] | "
        f"acc={metrics['accuracy']:.4f} | "
        f"cv_best={search.best_score_:.4f} | "
        f"{elapsed:.1f}s"
    )

    return metrics


def save_results(results: dict, model_name: str):
    """save results dict to json (convert numpy types)."""
    out_dir = cfg.results_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # convert numpy types for json serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    serializable = {}
    for clf_name, metrics in results.items():
        serializable[clf_name] = {
            k: convert(v) if not isinstance(v, dict) else {
                kk: convert(vv) for kk, vv in v.items()
            }
            for k, v in metrics.items()
        }

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=convert)

    logger.info(f"saved results to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="train classifiers on esm-2 embeddings"
    )
    parser.add_argument(
        "--model", type=str, default="esm2_8m",
        choices=list(cfg.models.keys()) + ["all"],
        help="which embedding model to use",
    )
    parser.add_argument(
        "--clf", nargs="+", default=None,
        choices=list(cfg.clf_params.keys()),
        help="which classifiers to train (default: all)",
    )
    args = parser.parse_args()

    model_names = list(cfg.models.keys()) if args.model == "all" else [args.model]
    clf_names = args.clf or list(cfg.clf_params.keys())

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL: {model_name}")
        logger.info(f"{'='*60}")

        # load embeddings
        try:
            x_train, y_train_str, _ = load_embeddings(model_name, "train")
            x_test, y_test_str, _ = load_embeddings(model_name, "test")
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        # encode labels
        y_train, le = encode_labels(y_train_str.tolist())
        y_test, _ = encode_labels(y_test_str.tolist(), le)

        logger.info(f"train: {x_train.shape}, test: {x_test.shape}")
        logger.info(f"classes: {le.classes_.tolist()}")

        # train each classifier
        results = {}
        for clf_name in clf_names:
            metrics = train_and_evaluate(
                x_train, y_train, x_test, y_test, clf_name
            )
            if metrics:
                results[clf_name] = metrics

        # save results
        save_results(results, model_name)

        # print summary table
        logger.info(f"\n{'─'*60}")
        logger.info(f"SUMMARY — {model_name}")
        logger.info(f"{'─'*60}")
        logger.info(f"{'classifier':<25} {'macro_f1':>10} {'accuracy':>10} {'avg_mcc':>10}")
        logger.info(f"{'─'*25} {'─'*10} {'─'*10} {'─'*10}")
        for clf_name, m in sorted(results.items(), key=lambda x: -x[1]["macro_f1"]):
            logger.info(
                f"{clf_name:<25} {m['macro_f1']:>10.4f} "
                f"{m['accuracy']:>10.4f} {m['avg_mcc']:>10.4f}"
            )


if __name__ == "__main__":
    main()
