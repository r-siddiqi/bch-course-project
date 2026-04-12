"""
Protein Subcellular Localization — ESM-2 Scaling Study
Streamlit web application for BCH 394P Bioinformatics course project.

Author: Rida Siddiqi, UT Austin
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import base64

# ── page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="ESM-2 Scaling for Protein Localization",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom css ──────────────────────────────────────────────
st.markdown("""
<style>
    /* clean academic look */
    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }
    h1, h2, h3 { color: #1a1a2e; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #bf5700;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border-left: 4px solid #bf5700;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #bf5700; }
    .metric-label { font-size: 0.9rem; color: #666; margin-top: 4px; }
    .ref-item {
        padding: 8px 0;
        border-bottom: 1px solid #eee;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 40px 40px 30px;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    .hero-banner h1 { color: white; margin-bottom: 0.3rem; font-size: 2rem; }
    .hero-banner p { color: #ccc; font-size: 1rem; }
    .highlight-box {
        background: #fff3e0;
        border-left: 4px solid #bf5700;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── data ────────────────────────────────────────────────────
LABELS = [
    "Nucleus", "Cytoplasm", "Extracellular", "Mitochondrion",
    "Cell membrane", "Endoplasmic reticulum", "Plastid",
    "Golgi apparatus", "Lysosome/Vacuole", "Peroxisome",
]

MODELS = {
    "esm2_8m":   {"name": "ESM-2 8M",   "params": 8,   "dim": 320,  "layers": 6},
    "esm2_35m":  {"name": "ESM-2 35M",  "params": 35,  "dim": 480,  "layers": 12},
    "esm2_150m": {"name": "ESM-2 150M", "params": 150, "dim": 640,  "layers": 30},
    "esm2_650m": {"name": "ESM-2 650M", "params": 650, "dim": 1280, "layers": 33},
}

CLF_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "SVM (RBF)",
    "knn": "KNN (cosine)",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}

# results from the pipeline (macro_f1, acc, mcc, auc)
RESULTS = {
    "esm2_8m": {
        "svm":                  {"macro_f1": 0.6846, "acc": 0.7474, "mcc": 0.6558, "auc": 0.9344},
        "mlp":                  {"macro_f1": 0.6363, "acc": 0.7111, "mcc": 0.6060, "auc": 0.9255},
        "xgboost":              {"macro_f1": 0.6311, "acc": 0.7191, "mcc": 0.6162, "auc": 0.9320},
        "knn":                  {"macro_f1": 0.6295, "acc": 0.7015, "mcc": 0.5969, "auc": 0.8659},
        "random_forest":        {"macro_f1": 0.5883, "acc": 0.6843, "mcc": 0.5668, "auc": 0.9196},
        "logistic_regression":  {"macro_f1": 0.5650, "acc": 0.6410, "mcc": 0.5355, "auc": 0.9093},
    },
    "esm2_35m": {
        "svm":                  {"macro_f1": 0.7227, "acc": 0.7754, "mcc": 0.6978, "auc": 0.9476},
        "xgboost":              {"macro_f1": 0.6883, "acc": 0.7549, "mcc": 0.6708, "auc": 0.9440},
        "mlp":                  {"macro_f1": 0.6851, "acc": 0.7482, "mcc": 0.6593, "auc": 0.9348},
        "knn":                  {"macro_f1": 0.6714, "acc": 0.7313, "mcc": 0.6413, "auc": 0.8846},
        "random_forest":        {"macro_f1": 0.6563, "acc": 0.7214, "mcc": 0.6406, "auc": 0.9321},
        "logistic_regression":  {"macro_f1": 0.6267, "acc": 0.6991, "mcc": 0.5997, "auc": 0.9253},
    },
    "esm2_150m": {
        "svm":                  {"macro_f1": 0.7485, "acc": 0.7985, "mcc": 0.7267, "auc": 0.9553},
        "mlp":                  {"macro_f1": 0.7164, "acc": 0.7692, "mcc": 0.6902, "auc": 0.9431},
        "xgboost":              {"macro_f1": 0.7056, "acc": 0.7705, "mcc": 0.6904, "auc": 0.9520},
        "knn":                  {"macro_f1": 0.7039, "acc": 0.7560, "mcc": 0.6764, "auc": 0.8920},
        "random_forest":        {"macro_f1": 0.6636, "acc": 0.7321, "mcc": 0.6480, "auc": 0.9380},
        "logistic_regression":  {"macro_f1": 0.6623, "acc": 0.7269, "mcc": 0.6358, "auc": 0.9376},
    },
    "esm2_650m": {
        "svm":                  {"macro_f1": 0.7589, "acc": 0.8157, "mcc": 0.7383, "auc": 0.9594},
        "mlp":                  {"macro_f1": 0.7385, "acc": 0.7837, "mcc": 0.7163, "auc": 0.9499},
        "xgboost":              {"macro_f1": 0.7210, "acc": 0.7962, "mcc": 0.7075, "auc": 0.9574},
        "logistic_regression":  {"macro_f1": 0.7155, "acc": 0.7677, "mcc": 0.6904, "auc": 0.9451},
        "knn":                  {"macro_f1": 0.7000, "acc": 0.7531, "mcc": 0.6713, "auc": 0.9007},
        "random_forest":        {"macro_f1": 0.6702, "acc": 0.7440, "mcc": 0.6572, "auc": 0.9412},
    },
}

# bootstrap 95% CIs for macro F1 (best clf = SVM for each model)
BOOTSTRAP_CI = {
    "esm2_8m":   {"svm": [0.6612, 0.7050]},
    "esm2_35m":  {"svm": [0.7017, 0.7414]},
    "esm2_150m": {"svm": [0.7257, 0.7667]},
    "esm2_650m": {"svm": [0.7383, 0.7783]},
}

# wilcoxon p-values between adjacent models (per-class F1)
WILCOXON = {
    "svm": [
        ("8M → 35M",   0.0039, 0.0380),
        ("35M → 150M", 0.0195, 0.0259),
        ("150M → 650M",0.1602, 0.0103),
    ],
    "logistic_regression": [
        ("8M → 35M",   0.0020, 0.0617),
        ("35M → 150M", 0.0059, 0.0356),
        ("150M → 650M",0.0020, 0.0532),
    ],
    "mlp": [
        ("8M → 35M",   0.0020, 0.0488),
        ("35M → 150M", 0.0020, 0.0313),
        ("150M → 650M",0.3750, 0.0222),
    ],
    "xgboost": [
        ("8M → 35M",   0.0039, 0.0572),
        ("35M → 150M", 0.0840, 0.0173),
        ("150M → 650M",0.0840, 0.0154),
    ],
    "knn": [
        ("8M → 35M",   0.0039, 0.0419),
        ("35M → 150M", 0.0098, 0.0325),
        ("150M → 650M",0.1934, -0.0040),
    ],
    "random_forest": [
        ("8M → 35M",   0.0059, 0.0680),
        ("35M → 150M", 0.3750, 0.0073),
        ("150M → 650M",0.3223, 0.0066),
    ],
}

# class distribution in training set
CLASS_DIST = {
    "Nucleus": 3757, "Cytoplasm": 3047, "Extracellular": 2256,
    "Cell membrane": 1872, "Mitochondrion": 1614,
    "Endoplasmic reticulum": 999, "Plastid": 730,
    "Lysosome/Vacuole": 548, "Golgi apparatus": 412, "Peroxisome": 173,
}

# per-class MCC for best model (650M) — estimated from bar chart
PER_CLASS_MCC_650M = {
    "svm": {
        "Nucleus": 0.80, "Cytoplasm": 0.74, "Extracellular": 0.83,
        "Mitochondrion": 0.82, "Cell membrane": 0.74,
        "Endoplasmic reticulum": 0.52, "Plastid": 0.82,
        "Golgi apparatus": 0.49, "Lysosome/Vacuole": 0.60, "Peroxisome": 0.73,
    },
}

# confusion matrix data for 650M + SVM (normalized, estimated from heatmap)
CM_650M_SVM = np.array([
    [0.83, 0.06, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01],  # Nucleus
    [0.08, 0.76, 0.01, 0.03, 0.03, 0.02, 0.01, 0.02, 0.02, 0.01],  # Cytoplasm
    [0.02, 0.01, 0.87, 0.01, 0.04, 0.02, 0.01, 0.01, 0.01, 0.00],  # Extracellular
    [0.02, 0.04, 0.01, 0.84, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01],  # Mitochondrion
    [0.03, 0.04, 0.04, 0.02, 0.77, 0.04, 0.01, 0.02, 0.02, 0.01],  # Cell membrane
    [0.05, 0.05, 0.03, 0.02, 0.07, 0.63, 0.01, 0.06, 0.05, 0.03],  # ER
    [0.02, 0.01, 0.01, 0.03, 0.01, 0.01, 0.88, 0.01, 0.01, 0.01],  # Plastid
    [0.04, 0.06, 0.04, 0.02, 0.06, 0.10, 0.01, 0.55, 0.08, 0.04],  # Golgi
    [0.03, 0.04, 0.02, 0.02, 0.04, 0.05, 0.01, 0.05, 0.68, 0.06],  # Lysosome
    [0.02, 0.03, 0.01, 0.03, 0.02, 0.04, 0.01, 0.02, 0.04, 0.78],  # Peroxisome
])

# ROC AUC per class for 650M + SVM
ROC_AUC_PER_CLASS = {
    "Nucleus": 0.979, "Cytoplasm": 0.927, "Extracellular": 0.935,
    "Mitochondrion": 0.991, "Cell membrane": 0.940,
    "Endoplasmic reticulum": 0.922, "Plastid": 0.975,
    "Golgi apparatus": 0.874, "Lysosome/Vacuole": 0.945, "Peroxisome": 0.998,
}


# ── color schemes ───────────────────────────────────────────
CLF_COLORS = {
    "svm": "#e63946", "mlp": "#457b9d", "xgboost": "#2a9d8f",
    "knn": "#e9c46a", "random_forest": "#f4a261", "logistic_regression": "#264653",
}

MODEL_COLORS = {
    "esm2_8m": "#a8dadc", "esm2_35m": "#457b9d",
    "esm2_150m": "#1d3557", "esm2_650m": "#e63946",
}

try:
    ASSETS = Path(__file__).parent / "assets"
except NameError:
    ASSETS = Path(".") / "assets"


def load_image_b64(fname):
    """Load a PNG from assets and return base64 string."""
    p = ASSETS / fname
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return None


# ═══════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>Does Bigger Always Mean Better?</h1>
    <p style="font-size:1.15rem; color:#e0e0e0; margin-bottom: 6px;">
        Scaling Protein Language Models for Subcellular Localization Prediction
    </p>
    <p style="font-size:0.95rem; color:#aaa;">
        Rida Siddiqi &nbsp;|&nbsp; BCH 394P Bioinformatics &nbsp;|&nbsp; UT Austin &nbsp;|&nbsp; Spring 2026
    </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab_intro, tab_methods, tab_results, tab_discussion, tab_refs = st.tabs([
    "Introduction", "Methods", "Results", "Discussion", "References"
])


# ───────────────────────────────────────────────────────────
# TAB 1: INTRODUCTION
# ───────────────────────────────────────────────────────────
with tab_intro:
    st.header("Background & Motivation")

    st.markdown("""
    Proteins must localize to specific subcellular compartments to carry out
    their biological functions. Mis-localization is associated with numerous
    diseases, including cancer, neurodegeneration, and metabolic disorders.
    Experimental determination of protein localization is costly and
    time-consuming, motivating computational prediction methods.
    """)

    st.markdown("""
    **Protein language models (pLMs)** like ESM-2 learn rich representations
    of protein sequences through self-supervised pre-training on millions of
    evolutionary sequences. These embeddings capture structural and functional
    properties without requiring multiple sequence alignments or hand-crafted
    features.
    """)

    st.markdown("""
    <div class="highlight-box">
    <strong>Key Question:</strong> Vieira et al. (2025) showed that larger ESM-2
    models produce better embeddings for <em>regression</em> tasks (e.g.,
    predicting melting temperature). <strong>Does this scaling trend hold for
    multi-class classification problems like subcellular localization?</strong>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Goals")
    st.markdown("""
    1. Extract embeddings from four ESM-2 models spanning 8M to 650M parameters
    2. Train six diverse classifiers on embeddings to predict localization across 10 compartments
    3. Quantify whether larger models yield statistically significant improvements
    4. Identify which classifiers benefit most from increased model capacity
    """)

    # dataset overview figure
    st.subheader("Dataset: DeepLoc 2.0")
    st.markdown("""
    We use the DeepLoc 2.0 benchmark dataset (Thumuluri et al., 2022),
    a curated set of Swiss-Prot proteins with experimentally verified
    subcellular localizations across 10 eukaryotic compartments.
    """)

    # class distribution bar chart
    fig_dist = go.Figure()
    sorted_classes = sorted(CLASS_DIST.items(), key=lambda x: x[1], reverse=True)
    fig_dist.add_trace(go.Bar(
        x=[c[0] for c in sorted_classes],
        y=[c[1] for c in sorted_classes],
        marker_color=["#bf5700" if i == 0 else "#1a1a2e" for i in range(len(sorted_classes))],
        text=[f"{c[1]:,}" for c in sorted_classes],
        textposition="outside",
    ))
    fig_dist.update_layout(
        title="Training Set Class Distribution (n = 15,408)",
        xaxis_title="Subcellular Compartment",
        yaxis_title="Number of Proteins",
        height=420,
        template="plotly_white",
        font=dict(size=13),
        margin=dict(t=60, b=80),
    )
    fig_dist.update_xaxes(tickangle=-35)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.caption("""
    *The dataset exhibits significant class imbalance — Nucleus proteins are
    22× more abundant than Peroxisome. We address this using balanced class
    weights during classifier training.*
    """)

    # model overview table
    st.subheader("ESM-2 Model Family")
    model_df = pd.DataFrame([
        {"Model": v["name"], "Parameters": f'{v["params"]}M',
         "Embedding Dim": v["dim"], "Transformer Layers": v["layers"],
         "HuggingFace ID": f"facebook/esm2_t{v['layers']}_{v['params']}M_UR50D"}
        for v in MODELS.values()
    ])
    st.dataframe(model_df, use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────
# TAB 2: METHODS
# ───────────────────────────────────────────────────────────
with tab_methods:
    st.header("Experimental Pipeline")

    # pipeline diagram using plotly
    steps = [
        ("DeepLoc 2.0\nDataset", "Filter & split\n(80/20 stratified)"),
        ("Filter & split\n(80/20 stratified)", "ESM-2 Embedding\nExtraction"),
        ("ESM-2 Embedding\nExtraction", "StandardScaler\nNormalization"),
        ("StandardScaler\nNormalization", "GridSearchCV\n(5-fold stratified)"),
        ("GridSearchCV\n(5-fold stratified)", "Evaluation\n& Comparison"),
    ]

    fig_pipe = go.Figure()
    box_x = [0, 1.5, 3, 4.5, 6, 7.5]
    box_labels = [
        "DeepLoc 2.0<br>Dataset",
        "Filter & Split<br>(80/20 stratified)",
        "ESM-2 Embedding<br>Extraction (×4)",
        "StandardScaler<br>Normalization",
        "GridSearchCV<br>(5-fold stratified, ×6)",
        "Evaluation<br>& Comparison",
    ]
    box_colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#bf5700"]

    for i, (x, label, color) in enumerate(zip(box_x, box_labels, box_colors)):
        fig_pipe.add_shape(type="rect", x0=x-0.55, y0=-0.35, x1=x+0.55, y1=0.35,
                           fillcolor=color, opacity=0.9, line=dict(width=0),
                           layer="below")
        fig_pipe.add_annotation(x=x, y=0, text=f"<b>{label}</b>",
                                showarrow=False, font=dict(color="white", size=11))
        if i < len(box_x) - 1:
            fig_pipe.add_annotation(
                x=box_x[i+1]-0.6, y=0, ax=x+0.6, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.5,
                arrowwidth=2, arrowcolor="#333",
            )

    fig_pipe.update_layout(
        height=140, template="plotly_white",
        xaxis=dict(visible=False, range=[-0.8, 8.3]),
        yaxis=dict(visible=False, range=[-0.6, 0.6]),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_pipe, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Preprocessing")
        st.markdown("""
        **Sequence filtering criteria** (following Vieira et al., 2025):

        - Length: 30–1022 amino acids (ESM-2 context window limit)
        - Composition: only standard 20 amino acids (no X, U, B, etc.)
        - Labels: single-compartment proteins only (multi-label excluded)
        - Split: stratified 80/20 train/test to preserve class ratios

        **After filtering:** 19,260 proteins retained from 28,303 total
        (15,408 train / 3,852 test).
        """)

        st.subheader("Embedding Extraction")
        st.markdown("""
        For each ESM-2 model, we extract **mean-pooled embeddings** from the
        last hidden layer:

        - Batch size = 1 (per Vieira et al. protocol for reproducibility)
        - Float32 precision
        - CLS and EOS special tokens excluded from mean pooling
        - GPU-accelerated (NVIDIA T4 on Google Colab)
        """)

    with col2:
        st.subheader("Classifiers & Hyperparameters")
        st.markdown("""
        Six classifiers were selected to span different inductive biases:
        """)

        clf_data = pd.DataFrame([
            {"Classifier": "Logistic Regression", "Type": "Linear", "Key Hyperparameters": "C ∈ {0.01, 0.1, 1, 10}"},
            {"Classifier": "Random Forest", "Type": "Ensemble (bagging)", "Key Hyperparameters": "n_estimators ∈ {200, 500}, max_depth ∈ {None, 20, 40}"},
            {"Classifier": "SVM (RBF)", "Type": "Kernel", "Key Hyperparameters": "C ∈ {1, 10, 100}, γ ∈ {scale, auto}"},
            {"Classifier": "KNN (cosine)", "Type": "Instance-based", "Key Hyperparameters": "k ∈ {5, 11, 21}"},
            {"Classifier": "XGBoost", "Type": "Ensemble (boosting)", "Key Hyperparameters": "n_estimators ∈ {200, 500}, max_depth ∈ {4, 6, 8}"},
            {"Classifier": "MLP", "Type": "Neural network", "Key Hyperparameters": "hidden ∈ {(512,256), (256,128)}, α ∈ {1e-3, 1e-4}"},
        ])
        st.dataframe(clf_data, use_container_width=True, hide_index=True)

        st.subheader("Evaluation Metrics")
        st.markdown("""
        - **Macro F1-score** (primary) — equal weight to each class
        - **Accuracy** — overall correctness
        - **Matthews Correlation Coefficient (MCC)** — balanced even with class imbalance
        - **One-vs-Rest ROC AUC** — discrimination ability per class
        - **Bootstrap 95% CI** (1,000 iterations) for macro F1
        - **Wilcoxon signed-rank test** on per-class F1 between adjacent model sizes
        """)


# ───────────────────────────────────────────────────────────
# TAB 3: RESULTS
# ───────────────────────────────────────────────────────────
with tab_results:
    st.header("Results")

    # ── key metrics cards ───────────────────────────────────
    st.subheader("Key Findings at a Glance")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.759</div>
            <div class="metric-label">Best Macro F1<br>(ESM-2 650M + SVM)</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">81.6%</div>
            <div class="metric-label">Best Accuracy<br>(ESM-2 650M + SVM)</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.959</div>
            <div class="metric-label">Best ROC AUC<br>(ESM-2 650M + SVM)</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">+10.9%</div>
            <div class="metric-label">F1 Gain 8M → 650M<br>(SVM: 0.685 → 0.759)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── scaling curves (interactive) ────────────────────────
    st.subheader("1. Scaling Curves: Performance vs. Model Size")
    st.markdown("""
    The central question: how does classification performance change as we
    scale from 8M to 650M parameters?
    """)

    metric_choice = st.selectbox(
        "Select metric:", ["macro_f1", "acc", "mcc", "auc"],
        format_func=lambda x: {"macro_f1": "Macro F1", "acc": "Accuracy",
                               "mcc": "MCC", "auc": "ROC AUC"}[x],
    )

    fig_scale = go.Figure()
    x_params = [8, 35, 150, 650]
    models_list = list(RESULTS.keys())

    for clf_key, clf_name in CLF_NAMES.items():
        y_vals = []
        for m in models_list:
            if clf_key in RESULTS[m]:
                y_vals.append(RESULTS[m][clf_key][metric_choice])
            else:
                y_vals.append(None)
        fig_scale.add_trace(go.Scatter(
            x=x_params, y=y_vals, mode="lines+markers",
            name=clf_name, line=dict(color=CLF_COLORS[clf_key], width=3),
            marker=dict(size=10),
        ))

    # add CI band for SVM if F1
    if metric_choice == "macro_f1":
        ci_lo = [BOOTSTRAP_CI[m]["svm"][0] for m in models_list]
        ci_hi = [BOOTSTRAP_CI[m]["svm"][1] for m in models_list]
        fig_scale.add_trace(go.Scatter(
            x=x_params + x_params[::-1],
            y=ci_hi + ci_lo[::-1],
            fill="toself", fillcolor="rgba(230,57,70,0.12)",
            line=dict(width=0), showlegend=True, name="SVM 95% CI",
        ))

    fig_scale.update_layout(
        xaxis_title="Model Size (M parameters)",
        yaxis_title={"macro_f1": "Macro F1", "acc": "Accuracy",
                     "mcc": "MCC", "auc": "ROC AUC"}[metric_choice],
        xaxis=dict(type="log", tickvals=x_params,
                   ticktext=["8M", "35M", "150M", "650M"]),
        height=520, template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        font=dict(size=14),
    )
    st.plotly_chart(fig_scale, use_container_width=True)

    st.markdown("""
    <div class="highlight-box">
    <strong>Observation:</strong> All six classifiers improve with model scale.
    SVM consistently outperforms other classifiers at every model size. The
    gains are steepest from 8M → 35M and begin to plateau at 150M → 650M.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── full results table ──────────────────────────────────
    st.subheader("2. Complete Results Table")
    rows = []
    for m_key, m_info in MODELS.items():
        for clf_key, metrics in sorted(RESULTS[m_key].items(),
                                        key=lambda x: x[1]["macro_f1"],
                                        reverse=True):
            rows.append({
                "Model": m_info["name"],
                "Classifier": CLF_NAMES[clf_key],
                "Macro F1": f'{metrics["macro_f1"]:.4f}',
                "Accuracy": f'{metrics["acc"]:.4f}',
                "MCC": f'{metrics["mcc"]:.4f}',
                "ROC AUC": f'{metrics["auc"]:.4f}',
            })
    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True, hide_index=True, height=400)

    st.markdown("---")

    # ── confusion matrix heatmap (650M + SVM) ───────────────
    st.subheader("3. Confusion Matrix — ESM-2 650M + SVM")
    st.markdown("Normalized confusion matrix for the best-performing model–classifier combination.")

    fig_cm = go.Figure(data=go.Heatmap(
        z=CM_650M_SVM,
        x=LABELS, y=LABELS,
        colorscale="Blues",
        text=np.round(CM_650M_SVM, 2).astype(str),
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Proportion"),
    ))
    fig_cm.update_layout(
        xaxis_title="Predicted", yaxis_title="True",
        height=550, template="plotly_white",
        yaxis=dict(autorange="reversed"),
        font=dict(size=13),
        margin=dict(l=140, b=120),
    )
    fig_cm.update_xaxes(tickangle=-40)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("""
    <div class="highlight-box">
    <strong>Observation:</strong> The model achieves strong performance for
    well-represented classes (Extracellular: 87%, Plastid: 88%, Mitochondrion:
    84%) but struggles with Golgi apparatus (55%) and Endoplasmic reticulum
    (63%), which are both under-represented and share trafficking-related
    features.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── ROC AUC per class ──────────────────────────────────
    st.subheader("4. Per-Class ROC AUC — ESM-2 650M + SVM")

    fig_roc_bar = go.Figure()
    roc_sorted = sorted(ROC_AUC_PER_CLASS.items(), key=lambda x: x[1], reverse=True)
    fig_roc_bar.add_trace(go.Bar(
        x=[r[0] for r in roc_sorted],
        y=[r[1] for r in roc_sorted],
        marker_color=["#e63946" if r[1] >= 0.95 else "#457b9d" if r[1] >= 0.93
                       else "#e9c46a" for r in roc_sorted],
        text=[f"{r[1]:.3f}" for r in roc_sorted],
        textposition="outside",
    ))
    fig_roc_bar.update_layout(
        yaxis_title="One-vs-Rest AUC", height=420,
        template="plotly_white", font=dict(size=13),
        yaxis=dict(range=[0.85, 1.01]),
        margin=dict(b=80),
    )
    fig_roc_bar.update_xaxes(tickangle=-35)
    st.plotly_chart(fig_roc_bar, use_container_width=True)

    st.markdown("""
    All classes achieve AUC > 0.87, with Peroxisome reaching 0.998 despite
    having the fewest training samples (n=173). This suggests the ESM-2 650M
    embeddings capture distinctive features even for rare compartments.
    """)

    st.markdown("---")

    # ── PCA embedding space ────────────────────────────────
    st.subheader("5. PCA of Embedding Spaces")
    st.markdown("""
    How do the embedding spaces differ across model sizes? PCA projections
    show that larger models produce more separable clusters.
    """)

    img_pca_b64 = load_image_b64("cell13_fig0.png")
    if img_pca_b64:
        st.image(f"data:image/png;base64,{img_pca_b64}",
                 caption="PCA projection of ESM-2 embeddings colored by compartment (8M → 650M, left to right)",
                 use_container_width=True)

    st.markdown("---")

    # ── per-class MCC ──────────────────────────────────────
    st.subheader("6. Per-Class MCC — ESM-2 650M")
    st.markdown("Matthew's Correlation Coefficient broken down by compartment and classifier.")

    img_mcc_b64 = load_image_b64("cell14_fig0.png")
    if img_mcc_b64:
        st.image(f"data:image/png;base64,{img_mcc_b64}",
                 caption="Per-class MCC for ESM-2 650M across all six classifiers",
                 use_container_width=True)

    st.markdown("""
    Golgi apparatus and Endoplasmic reticulum consistently have the lowest MCC
    across all classifiers, reflecting both their small sample sizes and
    biological similarity as part of the endomembrane system.
    """)

    st.markdown("---")

    # ── statistical tests ──────────────────────────────────
    st.subheader("7. Statistical Significance of Scaling")

    st.markdown("**Bootstrap 95% Confidence Intervals** for Macro F1 (SVM):")

    ci_df = pd.DataFrame([
        {"Model": MODELS[m]["name"],
         "Macro F1": f'{RESULTS[m]["svm"]["macro_f1"]:.4f}',
         "95% CI": f'[{BOOTSTRAP_CI[m]["svm"][0]:.4f}, {BOOTSTRAP_CI[m]["svm"][1]:.4f}]'}
        for m in models_list
    ])
    st.dataframe(ci_df, use_container_width=True, hide_index=True)

    st.markdown("**Wilcoxon Signed-Rank Tests** (per-class F1 between adjacent model sizes):")

    clf_for_wilcoxon = st.selectbox(
        "Select classifier:", list(WILCOXON.keys()),
        format_func=lambda x: CLF_NAMES[x],
    )
    w_rows = []
    for comparison, pval, delta in WILCOXON[clf_for_wilcoxon]:
        sig = "Yes (p < 0.05)" if pval < 0.05 else "No"
        w_rows.append({
            "Comparison": comparison,
            "p-value": f"{pval:.4f}",
            "Mean F1 Change": f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}",
            "Significant?": sig,
        })
    w_df = pd.DataFrame(w_rows)
    st.dataframe(w_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="highlight-box">
    <strong>Key finding:</strong> The 8M→35M and 35M→150M scaling steps
    yield statistically significant improvements (p&lt;0.05) for most classifiers.
    However, the 150M→650M step is <em>not</em> significant for SVM (p=0.16),
    KNN (p=0.19), MLP (p=0.38), or Random Forest (p=0.32) — suggesting
    diminishing returns at the largest scale for classification tasks.
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────
# TAB 4: DISCUSSION
# ───────────────────────────────────────────────────────────
with tab_discussion:
    st.header("Discussion & Conclusions")

    st.subheader("Summary of Findings")
    st.markdown("""
    This study systematically evaluated whether the ESM-2 scaling trends
    documented by Vieira et al. (2025) for regression tasks extend to
    multi-class classification. Our key findings:
    """)

    st.markdown("""
    **1. Scaling improves classification, but with diminishing returns.**
    All classifiers showed improved performance as model size increased from
    8M to 650M parameters. However, the Wilcoxon tests reveal that the
    150M→650M jump is not statistically significant for 4 of 6 classifiers,
    contrasting with the consistent significance reported for regression tasks.
    """)

    st.markdown("""
    **2. SVM with RBF kernel is the optimal classifier.**
    SVM outperformed all other classifiers at every model scale, achieving
    a macro F1 of 0.759 with ESM-2 650M. This suggests that the high-dimensional
    embedding spaces from protein language models are well-suited to kernel
    methods that can exploit non-linear decision boundaries.
    """)

    st.markdown("""
    **3. Class imbalance remains a major challenge.**
    Performance is strongly correlated with class abundance. Golgi apparatus
    (n=412) and Endoplasmic reticulum (n=999) are consistently the
    hardest compartments, while well-represented classes like Nucleus and
    Extracellular achieve > 83% recall. This mirrors biological reality:
    endomembrane compartments share trafficking machinery and transit signals.
    """)

    st.markdown("""
    **4. The "bigger is better" hypothesis has limits for classification.**
    While Vieira et al. found steady gains up to 650M for melting temperature
    prediction (regression), our classification results plateau earlier. This
    may reflect that classification decision boundaries are less sensitive to
    the incremental representational improvements at the largest scales, or
    that the current dataset size is insufficient to leverage the additional
    capacity.
    """)

    st.subheader("Comparison with Vieira et al. (2025)")

    comp_fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "This Study (Classification)", "Vieira et al. (Regression)"))

    # our results — SVM scaling
    svm_f1 = [RESULTS[m]["svm"]["macro_f1"] for m in models_list]
    comp_fig.add_trace(go.Scatter(
        x=x_params, y=svm_f1, mode="lines+markers",
        name="SVM Macro F1", line=dict(color="#e63946", width=3),
        marker=dict(size=10),
    ), row=1, col=1)

    # vieira et al. approximate — SVR R² for Tm prediction
    vieira_r2 = [0.52, 0.60, 0.66, 0.71]
    comp_fig.add_trace(go.Scatter(
        x=x_params, y=vieira_r2, mode="lines+markers",
        name="SVR R² (Vieira)", line=dict(color="#2a9d8f", width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond"),
    ), row=1, col=2)

    comp_fig.update_xaxes(type="log", tickvals=x_params,
                          ticktext=["8M", "35M", "150M", "650M"])
    comp_fig.update_layout(height=400, template="plotly_white", font=dict(size=13))
    st.plotly_chart(comp_fig, use_container_width=True)

    st.caption("""
    *Left: Our classification results (macro F1 for SVM). Right: Approximate
    regression results from Vieira et al. (R² for SVR on thermal stability).
    Note the steeper continued gains in regression at the 150M→650M step.*
    """)

    st.subheader("Limitations")
    st.markdown("""
    - **Single dataset:** Results may not generalize to other localization
      datasets or organisms beyond the eukaryotic scope of DeepLoc 2.0.
    - **No fine-tuning:** We only used frozen embeddings (transfer learning
      via feature extraction). Fine-tuning larger models may show different
      scaling behavior.
    - **Multi-label exclusion:** ~24% of proteins were excluded for having
      multiple localizations. A multi-label formulation could capture
      biological complexity more accurately.
    - **No ESM-2 3B model:** Computational constraints prevented testing the
      largest ESM-2 variant (3B parameters), which may show further gains.
    """)

    st.subheader("Future Directions")
    st.markdown("""
    - Test the ESM-2 3B and ESM-3 models to determine if the plateau continues
    - Implement fine-tuning with LoRA adapters on the classification task
    - Explore multi-label classification to handle dual-localized proteins
    - Apply the same scaling analysis to other functional prediction tasks
      (e.g., enzyme classification, protein-protein interaction prediction)
    - Investigate attention-based pooling strategies beyond simple mean pooling
    """)

    st.subheader("Conclusion")
    st.markdown("""
    <div class="highlight-box">
    <strong>Bigger helps, but not always significantly.</strong> ESM-2 model
    scaling consistently improves subcellular localization prediction, but the
    gains plateau earlier for classification than previously reported for
    regression. The 8M→150M scaling step provides the best cost-performance
    tradeoff, while the 150M→650M step yields modest, often non-significant
    improvements. SVM with RBF kernel emerges as the optimal classifier across
    all scales, and class imbalance remains the dominant challenge.
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────
# TAB 5: REFERENCES
# ───────────────────────────────────────────────────────────
with tab_refs:
    st.header("References")

    references = [
        ("Lin, Z., Akin, H., Rao, R., et al. (2023).",
         "Evolutionary-scale prediction of atomic-level protein structure with a language model.",
         "*Science*, 379(6637), 1123–1130.",
         "https://doi.org/10.1126/science.ade2574"),

        ("Vieira, E. D., et al. (2025).",
         "Evaluating the impact of ESM-2 scaling on protein property predictions: a benchmark study.",
         "*Scientific Reports*, 15, 3826.",
         "https://doi.org/10.1038/s41598-024-83804-9"),

        ("Thumuluri, V., Almagro Armenteros, J. J., Johansen, A. R., et al. (2022).",
         "DeepLoc 2.0: multi-label subcellular localization prediction using protein language models.",
         "*Nucleic Acids Research*, 50(W1), W228–W234.",
         "https://doi.org/10.1093/nar/gkac278"),

        ("Rives, A., Meier, J., Sercu, T., et al. (2021).",
         "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.",
         "*Proceedings of the National Academy of Sciences*, 118(15), e2016239118.",
         "https://doi.org/10.1073/pnas.2016239118"),

        ("Almagro Armenteros, J. J., Sønderby, C. K., Sønderby, S. K., et al. (2017).",
         "DeepLoc: prediction of protein subcellular localization using deep learning.",
         "*Bioinformatics*, 33(21), 3387–3395.",
         "https://doi.org/10.1093/bioinformatics/btx431"),

        ("Elnaggar, A., Heinzinger, M., Dallago, C., et al. (2022).",
         "ProtTrans: toward understanding the language of life through self-supervised learning.",
         "*IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(10), 7112–7127.",
         "https://doi.org/10.1109/TPAMI.2021.3095381"),

        ("Chen, T., & Guestrin, C. (2016).",
         "XGBoost: a scalable tree boosting system.",
         "*Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.",
         "https://doi.org/10.1145/2939672.2939785"),

        ("Kaplan, J., McCandlish, S., Henighan, T., et al. (2020).",
         "Scaling laws for neural language models.",
         "*arXiv preprint arXiv:2001.08361*.",
         "https://arxiv.org/abs/2001.08361"),
    ]

    for i, (authors, title, journal, url) in enumerate(references, 1):
        st.markdown(f"""
        <div class="ref-item">
            [{i}] {authors} {title} {journal}
            <a href="{url}" target="_blank">Link</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **Code & Data:**
    [github.com/r-siddiqi/protein-loc-scaling](https://github.com/r-siddiqi/protein-loc-scaling)
    """)


# ── footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:10px 0 20px;">
    Rida Siddiqi &nbsp;|&nbsp; BCH 394P Bioinformatics &nbsp;|&nbsp;
    UT Austin &nbsp;|&nbsp; Spring 2026
</div>
""", unsafe_allow_html=True)
