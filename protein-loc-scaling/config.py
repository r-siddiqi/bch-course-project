"""
project configuration — model names, paths, hyperparameters.
"""

from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent
data_dir = root / "data"
emb_dir = root / "embeddings"
results_dir = root / "results"

# ── esm-2 model variants (huggingface ids) ─────────────────────────
# maps friendly name → (hf model id, embedding dim)
models = {
    "esm2_8m":   ("facebook/esm2_t6_8M_UR50D",    320),
    "esm2_35m":  ("facebook/esm2_t12_35M_UR50D",   480),
    "esm2_150m": ("facebook/esm2_t30_150M_UR50D",   640),
    "esm2_650m": ("facebook/esm2_t33_650M_UR50D",  1280),
}

# ── deeploc 2.0 class labels (10 compartments) ────────────────────
labels = [
    "Nucleus",
    "Cytoplasm",
    "Extracellular",
    "Mitochondrion",
    "Cell membrane",
    "Endoplasmic reticulum",
    "Plastid",
    "Golgi apparatus",
    "Lysosome/Vacuole",
    "Peroxisome",
]
n_classes = len(labels)

# ── sequence constraints ───────────────────────────────────────────
max_seq_len = 1022   # esm-2 positional encoding limit
min_seq_len = 30     # discard very short fragments

# ── classifier hyperparameter grids ────────────────────────────────
clf_params = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": 2000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "multi_class": "multinomial",
    },
    "random_forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 20, 50],
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "svm": {
        "C": [0.1, 1.0, 10.0],
        "kernel": "rbf",
        "class_weight": "balanced",
        "decision_function_shape": "ovr",
    },
    "knn": {
        "n_neighbors": [3, 5, 11, 21],
        "weights": "distance",
        "metric": "cosine",
    },
    "xgboost": {
        "n_estimators": [100, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1],
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
    },
    "mlp": {
        "hidden_layer_sizes": [(256,), (256, 128), (512, 256)],
        "activation": "relu",
        "learning_rate_init": [0.001, 0.0001],
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.1,
    },
}

# ── evaluation ─────────────────────────────────────────────────────
n_bootstrap = 1000       # bootstrap iterations for ci
ci_level = 0.95          # confidence interval level
random_state = 42
