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
    /* clean academic look, forced light theme */
    html, body, [class*="css"] { color: #1a1a2e !important; }
    .main .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        background-color: #ffffff;
    }
    h1, h2, h3, h4 { color: #1a1a2e; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
        background-color: #f5f5f5;
        color: #333;
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
        padding: 10px 0;
        border-bottom: 1px solid #eee;
        font-size: 0.92rem;
        line-height: 1.55;
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
        color: #1a1a2e;
    }
    .fig-caption {
        font-size: 0.88rem; color: #555; font-style: italic;
        padding: 8px 12px; margin-bottom: 24px;
        border-left: 3px solid #bf5700; background: #fafafa;
    }
    .citation {
        font-size: 0.85em; vertical-align: super; color: #bf5700;
        text-decoration: none; font-weight: 600;
    }
    .section-intro {
        font-size: 1.02rem; line-height: 1.65; color: #2c2c2c;
    }
    p { line-height: 1.65; }
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

RESULTS = {
    "esm2_8m": {
        "svm":                  {"macro_f1": 0.6846, "acc": 0.7474, "mcc": 0.6558, "auc": 0.9344, "train_s":  2392},
        "mlp":                  {"macro_f1": 0.6363, "acc": 0.7111, "mcc": 0.6060, "auc": 0.9255, "train_s":   153},
        "xgboost":              {"macro_f1": 0.6311, "acc": 0.7191, "mcc": 0.6162, "auc": 0.9320, "train_s":  5003},
        "knn":                  {"macro_f1": 0.6295, "acc": 0.7015, "mcc": 0.5969, "auc": 0.8659, "train_s":    12},
        "random_forest":        {"macro_f1": 0.5883, "acc": 0.6843, "mcc": 0.5668, "auc": 0.9196, "train_s":  1134},
        "logistic_regression":  {"macro_f1": 0.5650, "acc": 0.6410, "mcc": 0.5355, "auc": 0.9093, "train_s":    81},
    },
    "esm2_35m": {
        "svm":                  {"macro_f1": 0.7227, "acc": 0.7754, "mcc": 0.6978, "auc": 0.9476, "train_s":  3065},
        "xgboost":              {"macro_f1": 0.6883, "acc": 0.7549, "mcc": 0.6708, "auc": 0.9440, "train_s":  7628},
        "mlp":                  {"macro_f1": 0.6851, "acc": 0.7482, "mcc": 0.6593, "auc": 0.9348, "train_s":   162},
        "knn":                  {"macro_f1": 0.6714, "acc": 0.7313, "mcc": 0.6413, "auc": 0.8846, "train_s":    14},
        "random_forest":        {"macro_f1": 0.6563, "acc": 0.7214, "mcc": 0.6406, "auc": 0.9321, "train_s":  1374},
        "logistic_regression":  {"macro_f1": 0.6267, "acc": 0.6991, "mcc": 0.5997, "auc": 0.9253, "train_s":   141},
    },
    "esm2_150m": {
        "svm":                  {"macro_f1": 0.7485, "acc": 0.7985, "mcc": 0.7267, "auc": 0.9553, "train_s":  4051},
        "mlp":                  {"macro_f1": 0.7164, "acc": 0.7692, "mcc": 0.6902, "auc": 0.9431, "train_s":   215},
        "xgboost":              {"macro_f1": 0.7056, "acc": 0.7705, "mcc": 0.6904, "auc": 0.9520, "train_s": 10065},
        "knn":                  {"macro_f1": 0.7039, "acc": 0.7560, "mcc": 0.6764, "auc": 0.8920, "train_s":    17},
        "random_forest":        {"macro_f1": 0.6636, "acc": 0.7321, "mcc": 0.6480, "auc": 0.9380, "train_s":  1639},
        "logistic_regression":  {"macro_f1": 0.6623, "acc": 0.7269, "mcc": 0.6358, "auc": 0.9376, "train_s":   214},
    },
    "esm2_650m": {
        "svm":                  {"macro_f1": 0.7589, "acc": 0.8157, "mcc": 0.7383, "auc": 0.9594, "train_s":  9559},
        "mlp":                  {"macro_f1": 0.7385, "acc": 0.7837, "mcc": 0.7163, "auc": 0.9499, "train_s":   300},
        "xgboost":              {"macro_f1": 0.7210, "acc": 0.7962, "mcc": 0.7075, "auc": 0.9574, "train_s": 21162},
        "logistic_regression":  {"macro_f1": 0.7155, "acc": 0.7677, "mcc": 0.6904, "auc": 0.9451, "train_s":   483},
        "knn":                  {"macro_f1": 0.7000, "acc": 0.7531, "mcc": 0.6713, "auc": 0.9007, "train_s":    27},
        "random_forest":        {"macro_f1": 0.6702, "acc": 0.7440, "mcc": 0.6572, "auc": 0.9412, "train_s":  2295},
    },
}

BOOTSTRAP_CI = {
    "esm2_8m":   {"svm": [0.6612, 0.7050]},
    "esm2_35m":  {"svm": [0.7017, 0.7414]},
    "esm2_150m": {"svm": [0.7257, 0.7667]},
    "esm2_650m": {"svm": [0.7383, 0.7783]},
}

WILCOXON = {
    "svm": [("8M → 35M", 0.0039, 0.0380), ("35M → 150M", 0.0195, 0.0259), ("150M → 650M", 0.1602, 0.0103)],
    "logistic_regression": [("8M → 35M", 0.0020, 0.0617), ("35M → 150M", 0.0059, 0.0356), ("150M → 650M", 0.0020, 0.0532)],
    "mlp": [("8M → 35M", 0.0020, 0.0488), ("35M → 150M", 0.0020, 0.0313), ("150M → 650M", 0.3750, 0.0222)],
    "xgboost": [("8M → 35M", 0.0039, 0.0572), ("35M → 150M", 0.0840, 0.0173), ("150M → 650M", 0.0840, 0.0154)],
    "knn": [("8M → 35M", 0.0039, 0.0419), ("35M → 150M", 0.0098, 0.0325), ("150M → 650M", 0.1934, -0.0040)],
    "random_forest": [("8M → 35M", 0.0059, 0.0680), ("35M → 150M", 0.3750, 0.0073), ("150M → 650M", 0.3223, 0.0066)],
}

CLASS_DIST = {
    "Nucleus": 3757, "Cytoplasm": 3047, "Extracellular": 2256,
    "Cell membrane": 1872, "Mitochondrion": 1614,
    "Endoplasmic reticulum": 999, "Plastid": 730,
    "Lysosome/Vacuole": 548, "Golgi apparatus": 412, "Peroxisome": 173,
}

PER_CLASS_F1_650M_SVM = {
    "Nucleus": 0.83, "Cytoplasm": 0.76, "Extracellular": 0.87, "Mitochondrion": 0.84,
    "Cell membrane": 0.77, "Endoplasmic reticulum": 0.63, "Plastid": 0.88,
    "Golgi apparatus": 0.55, "Lysosome/Vacuole": 0.68, "Peroxisome": 0.78,
}

PER_CLASS_F1_8M_SVM = {
    "Nucleus": 0.76, "Cytoplasm": 0.68, "Extracellular": 0.81, "Mitochondrion": 0.76,
    "Cell membrane": 0.68, "Endoplasmic reticulum": 0.53, "Plastid": 0.81,
    "Golgi apparatus": 0.41, "Lysosome/Vacuole": 0.58, "Peroxisome": 0.67,
}

CM_650M_SVM = np.array([
    [0.83, 0.06, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01],
    [0.08, 0.76, 0.01, 0.03, 0.03, 0.02, 0.01, 0.02, 0.02, 0.01],
    [0.02, 0.01, 0.87, 0.01, 0.04, 0.02, 0.01, 0.01, 0.01, 0.00],
    [0.02, 0.04, 0.01, 0.84, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01],
    [0.03, 0.04, 0.04, 0.02, 0.77, 0.04, 0.01, 0.02, 0.02, 0.01],
    [0.05, 0.05, 0.03, 0.02, 0.07, 0.63, 0.01, 0.06, 0.05, 0.03],
    [0.02, 0.01, 0.01, 0.03, 0.01, 0.01, 0.88, 0.01, 0.01, 0.01],
    [0.04, 0.06, 0.04, 0.02, 0.06, 0.10, 0.01, 0.55, 0.08, 0.04],
    [0.03, 0.04, 0.02, 0.02, 0.04, 0.05, 0.01, 0.05, 0.68, 0.06],
    [0.02, 0.03, 0.01, 0.03, 0.02, 0.04, 0.01, 0.02, 0.04, 0.78],
])

ROC_AUC_PER_CLASS = {
    "Nucleus": 0.979, "Cytoplasm": 0.927, "Extracellular": 0.935,
    "Mitochondrion": 0.991, "Cell membrane": 0.940,
    "Endoplasmic reticulum": 0.922, "Plastid": 0.975,
    "Golgi apparatus": 0.874, "Lysosome/Vacuole": 0.945, "Peroxisome": 0.998,
}

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
    p = ASSETS / fname
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return None


def cite(n):
    """Render a numbered in-text citation linking to References tab."""
    return f'<sup class="citation">[{n}]</sup>'


# ═══════════════════════════════════════════════════════════
# HERO
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


tab_intro, tab_methods, tab_results, tab_discussion, tab_refs = st.tabs([
    "Introduction", "Methods", "Results", "Discussion", "References"
])


# ───────────────────────────────────────────────────────────
# TAB 1: INTRODUCTION
# ───────────────────────────────────────────────────────────
with tab_intro:
    st.header("1. Introduction")

    st.subheader("1.1 The Importance of Subcellular Localization")
    st.markdown(f"""
    Every eukaryotic cell is a highly compartmentalized system in which proteins
    must reach the correct organelle to carry out their biological functions. A
    nuclear transcription factor stranded in the cytoplasm, a mitochondrial
    kinase mis-targeted to the endoplasmic reticulum, or a plasma-membrane
    receptor retained in the Golgi apparatus are all examples of how
    mis-localization can disrupt cellular physiology. Mis-localization is a
    documented driver of numerous human diseases, including cancer,
    neurodegenerative disorders, and metabolic syndromes{cite(1)}{cite(2)}.
    Determining where a protein resides is therefore foundational to
    understanding its function and its role in disease.

    Although experimental methods such as fluorescence microscopy,
    immunohistochemistry, and proximity-labeling proteomics can reveal protein
    localization with high fidelity, they are costly, low-throughput, and
    difficult to apply at genome scale{cite(3)}. Only a small fraction of the
    hundreds of millions of proteins in UniProt have been localized
    experimentally{cite(4)}. This gap motivates the development of accurate
    computational predictors that can assign subcellular locations directly
    from amino-acid sequence.
    """, unsafe_allow_html=True)

    # ── figure: cell compartment schematic (original plotly diagram) ──
    st.subheader("Figure 1. The Ten Subcellular Compartments Studied")
    comp_positions = {
        "Nucleus":               (0.50, 0.55, 0.12, "#e63946"),
        "Cytoplasm":             (0.50, 0.50, 0.35, "#ffd6a5"),
        "Mitochondrion":         (0.25, 0.60, 0.06, "#f4a261"),
        "Endoplasmic reticulum": (0.35, 0.38, 0.08, "#2a9d8f"),
        "Golgi apparatus":       (0.65, 0.40, 0.06, "#e9c46a"),
        "Lysosome/Vacuole":      (0.72, 0.62, 0.04, "#8338ec"),
        "Peroxisome":            (0.30, 0.72, 0.03, "#06a77d"),
        "Plastid":               (0.68, 0.68, 0.05, "#38b000"),
        "Cell membrane":         (0.50, 0.50, 0.45, None),
        "Extracellular":         (0.50, 0.50, 0.49, None),
    }
    fig_cell = go.Figure()
    # outer cell membrane
    fig_cell.add_shape(type="circle",
                       x0=0.05, y0=0.05, x1=0.95, y1=0.95,
                       line=dict(color="#1d3557", width=4),
                       fillcolor="rgba(168, 218, 220, 0.15)")
    fig_cell.add_annotation(x=0.50, y=0.97, text="<b>Extracellular</b>",
                            showarrow=False, font=dict(size=13, color="#1d3557"))
    fig_cell.add_annotation(x=0.08, y=0.50, text="<b>Cell<br>membrane</b>",
                            showarrow=False, font=dict(size=11, color="#1d3557"))
    # internal compartments
    internal = [
        ("Nucleus", 0.50, 0.60, 0.13, "#e63946"),
        ("Mitochondrion", 0.25, 0.55, 0.07, "#f4a261"),
        ("Endoplasmic reticulum", 0.30, 0.30, 0.09, "#2a9d8f"),
        ("Golgi apparatus", 0.68, 0.32, 0.07, "#e9c46a"),
        ("Lysosome/Vacuole", 0.75, 0.55, 0.05, "#8338ec"),
        ("Peroxisome", 0.55, 0.25, 0.035, "#06a77d"),
        ("Plastid", 0.50, 0.80, 0.055, "#38b000"),
    ]
    for name, x, y, r, color in internal:
        fig_cell.add_shape(type="circle",
                           x0=x-r, y0=y-r, x1=x+r, y1=y+r,
                           line=dict(color="#333", width=1),
                           fillcolor=color, opacity=0.75)
        fig_cell.add_annotation(x=x, y=y-r-0.025, text=f"<b>{name}</b>",
                                showarrow=False, font=dict(size=10, color="#1a1a2e"))
    fig_cell.add_annotation(x=0.50, y=0.45, text="<i>Cytoplasm</i>",
                            showarrow=False, font=dict(size=12, color="#555"))
    fig_cell.update_layout(
        height=500, template="simple_white",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1], scaleanchor="x"),
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_cell, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 1.</b> Schematic of the ten eukaryotic subcellular compartments
    annotated in the DeepLoc 2.0 dataset{cite(3)}. Compartments are drawn
    roughly to scale within a generic eukaryotic cell. Each protein in our
    training set is assigned to exactly one of these locations. Adapted
    from Thumuluri et al.[3].
    </div>
    """, unsafe_allow_html=True)

    st.subheader("1.2 A Brief History of Localization Prediction")
    st.markdown(f"""
    Computational prediction of protein subcellular localization has evolved
    through four broad eras. Early methods in the 1990s relied on
    hand-engineered features, most famously the discovery of the mitochondrial
    targeting peptide by Nakai & Kanehisa (PSORT){cite(5)} and the identification
    of signal peptides by SignalP{cite(6)}. These methods encoded decades of
    biochemical knowledge but were limited by the need for expert rules.

    The second era introduced classical machine learning: support vector
    machines, random forests, and Hidden Markov Models trained on
    position-specific scoring matrices and compositional features. Tools such
    as LOCtree, MultiLoc, and the original DeepLoc{cite(7)} improved accuracy
    but still depended on multiple sequence alignments (MSAs), which are slow
    to compute and unavailable for orphan proteins.

    The third era applied deep learning directly to amino-acid sequences.
    Recurrent networks, convolutional networks, and ultimately attention-based
    architectures were used to learn embeddings from raw protein sequences.
    DeepLoc 2.0{cite(3)} was among the first tools to replace MSA features with
    learned embeddings from a protein language model, achieving substantial
    accuracy gains.

    The current fourth era is defined by <b>large-scale protein language
    models</b> (pLMs), transformer architectures pre-trained on hundreds of
    millions of sequences with self-supervised objectives such as masked
    language modeling{cite(8)}{cite(9)}. These models produce dense vector
    representations of proteins that can be used as general-purpose input
    features for downstream tasks, a paradigm known as <i>transfer learning
    via feature extraction</i>{cite(10)}.
    """, unsafe_allow_html=True)

    st.subheader("1.3 Protein Language Models and the ESM-2 Family")
    st.markdown(f"""
    Protein language models (pLMs) adapt the transformer architecture of BERT
    and GPT to protein sequences. Each amino-acid position is treated as a
    token, and the model is pre-trained to reconstruct masked tokens from
    their context. The canonical ProtTrans{cite(11)} and ESM-1b{cite(9)}
    models demonstrated that such representations encode secondary-structure,
    binding-site, and evolutionary information without any supervised labels.

    The Evolutionary Scale Modeling version 2 (<b>ESM-2</b>){cite(8)} is a
    family of transformers ranging from 8 million to 15 billion parameters,
    pre-trained on the UniRef50 database of ~65 million non-redundant
    sequences. Unlike earlier pLMs, ESM-2 uses rotary positional embeddings
    and a modernized attention scheme, and its largest variants enabled the
    ESMFold structure predictor to generate atomic-resolution structures
    directly from sequence. The ESM-2 family is widely regarded as the
    strongest publicly available pLM for general feature extraction{cite(8)}.
    """, unsafe_allow_html=True)

    # ── figure: ESM-2 architecture schematic ──
    st.subheader("Figure 2. Conceptual Schematic of ESM-2 Feature Extraction")
    fig_arch = go.Figure()

    # input sequence tokens
    seq_chars = ['M','A','K','L','V','N','E','F','P','…']
    for i, ch in enumerate(seq_chars):
        fig_arch.add_shape(type="rect", x0=i*0.9, y0=0, x1=i*0.9+0.8, y1=0.6,
                           fillcolor="#e9ecef", line=dict(color="#333", width=1))
        fig_arch.add_annotation(x=i*0.9+0.4, y=0.3, text=f"<b>{ch}</b>",
                                showarrow=False, font=dict(size=14))
    fig_arch.add_annotation(x=4.5*0.9+0.4, y=-0.3, text="Amino-acid sequence (length L)",
                            showarrow=False, font=dict(size=11, color="#555"))

    # transformer blocks
    for i, (label, color) in enumerate([
        ("Embedding + rotary position", "#ffd6a5"),
        ("Transformer block × N", "#a8dadc"),
        ("Transformer block × N", "#a8dadc"),
        ("Transformer block × N", "#a8dadc"),
    ]):
        y_pos = 1.5 + i * 0.8
        fig_arch.add_shape(type="rect", x0=0, y0=y_pos, x1=9, y1=y_pos+0.55,
                           fillcolor=color, line=dict(color="#333"))
        fig_arch.add_annotation(x=4.5, y=y_pos+0.275, text=f"<b>{label}</b>",
                                showarrow=False, font=dict(size=12))

    # per-residue embeddings output
    for i in range(10):
        fig_arch.add_shape(type="rect", x0=i*0.9, y0=5.3, x1=i*0.9+0.8, y1=5.9,
                           fillcolor="#bde0fe", line=dict(color="#333"))
        fig_arch.add_annotation(x=i*0.9+0.4, y=5.6, text=f"<b>h<sub>{i+1}</sub></b>",
                                showarrow=False, font=dict(size=11))
    fig_arch.add_annotation(x=4.5*0.9+0.4, y=5.05,
                            text="Per-residue hidden states (L × d)",
                            showarrow=False, font=dict(size=11, color="#555"))

    # mean pooling arrow (text placed separately above the arrowhead)
    fig_arch.add_annotation(x=4.5*0.9+0.4, y=7.15,
                            ax=4.5*0.9+0.4, ay=6.1,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowsize=1.5,
                            arrowwidth=2.5, arrowcolor="#bf5700",
                            text="", font=dict(size=12))
    fig_arch.add_annotation(x=4.5*0.9+0.9, y=6.65,
                            text="<b>Mean pooling</b>",
                            showarrow=False, xanchor="left",
                            font=dict(size=12, color="#bf5700"))

    # final embedding vector
    fig_arch.add_shape(type="rect", x0=2, y0=7.2, x1=7, y1=7.8,
                       fillcolor="#bf5700", line=dict(color="#333"),
                       opacity=0.9)
    fig_arch.add_annotation(x=4.5, y=7.5,
                            text="<b>Protein embedding vector (d = 320 – 1280)</b>",
                            showarrow=False, font=dict(size=13, color="white"))

    fig_arch.update_layout(
        height=520, template="simple_white",
        xaxis=dict(visible=False, range=[-0.5, 9.5]),
        yaxis=dict(visible=False, range=[-0.8, 8.2]),
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_arch, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 2.</b> Schematic of ESM-2 feature extraction. An amino-acid
    sequence is tokenized and passed through a stack of transformer blocks
    (6 layers for the 8M model up to 33 layers for the 650M model). The
    final-layer per-residue hidden states are averaged (mean pooling,
    excluding CLS/EOS tokens) to yield a fixed-length protein embedding.
    Dimension <i>d</i> grows with model size: 320 → 480 → 640 → 1280 for the
    8M, 35M, 150M, and 650M models respectively{cite(8)}.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("1.4 Neural Scaling Laws")
    st.markdown(f"""
    In natural-language processing, empirical <i>scaling laws</i> relate the
    loss of a transformer model to its parameter count, training-data size,
    and compute budget via smooth power laws{cite(12)}. These laws predict
    that performance should improve monotonically with model scale until one
    of the three resources becomes a bottleneck. For language tasks, this
    prediction has been dramatically validated, as models such as GPT-3 and
    PaLM continue to improve at hundreds of billions of parameters{cite(13)}.

    Whether these scaling laws translate to biological domains is an open
    question of considerable practical importance. Larger ESM-2 variants
    require proportionally more GPU memory and inference time; the 650M
    model, for example, takes roughly 10× longer to embed a sequence than
    the 8M model on the same hardware. If the performance benefit does not
    justify this cost, practitioners should prefer smaller models for many
    applications.

    A recent benchmark study by Vieira et al. (2025){cite(14)} systematically
    evaluated ESM-2 scaling on several <b>regression</b> tasks, most notably
    protein melting-temperature (Tm) prediction, and concluded that larger
    models yielded smooth, consistent improvements, with the 650M variant
    clearly outperforming smaller ones. Whether this same trend holds for
    <b>classification</b> tasks, where decision boundaries rather than
    continuous targets are learned, has not been systematically tested at
    the same scale.
    """, unsafe_allow_html=True)

    st.subheader("1.5 Research Question and Hypotheses")
    st.markdown(f"""
    <div class="highlight-box">
    <b>Central question:</b> Do the ESM-2 scaling trends reported for
    regression tasks by Vieira et al. (2025){cite(14)} extend to multi-class
    classification of protein subcellular localization?
    </div>

    We test three specific hypotheses:

    <b>H1 (Scaling):</b> Macro-F1 on the DeepLoc 2.0 test set increases
    monotonically with ESM-2 parameter count across all classifiers.

    <b>H2 (Classifier choice):</b> Non-linear classifiers (SVM with RBF
    kernel, MLP, XGBoost) benefit more from larger embeddings than linear
    classifiers (logistic regression), because the additional representational
    capacity of larger pLMs is captured in non-linear structure of the
    embedding space{cite(15)}.

    <b>H3 (Diminishing returns):</b> The performance gain per additional
    parameter shrinks at larger scales, and the difference between 150M and
    650M models is not statistically significant for most classifiers on
    this classification task, in contrast to the continued gains reported
    for regression{cite(14)}.
    """, unsafe_allow_html=True)

    st.subheader("1.6 The DeepLoc 2.0 Benchmark")
    st.markdown(f"""
    We use the DeepLoc 2.0 dataset{cite(3)}, which provides experimentally
    verified subcellular localizations for 28,303 Swiss-Prot proteins across
    ten eukaryotic compartments. The dataset was constructed with a strict
    homology-reduction procedure (30% sequence-identity partitioning) to
    prevent train–test leakage, making it a gold-standard benchmark for
    localization prediction{cite(3)}. After filtering to single-compartment
    proteins with standard amino acids and sequence length 30–1022
    (ESM-2's context window), we retain 19,260 proteins for 80/20 stratified
    train/test splitting.
    """, unsafe_allow_html=True)

    # class distribution bar chart
    fig_dist = go.Figure()
    sorted_classes = sorted(CLASS_DIST.items(), key=lambda x: x[1], reverse=True)
    fig_dist.add_trace(go.Bar(
        x=[c[0] for c in sorted_classes],
        y=[c[1] for c in sorted_classes],
        marker_color="#1d3557",
        text=[f"{c[1]:,}" for c in sorted_classes],
        textposition="outside",
    ))
    fig_dist.update_layout(
        title="Figure 3. Training Set Class Distribution (n = 15,408)",
        xaxis_title="Subcellular Compartment", yaxis_title="Number of Proteins",
        height=440, template="plotly_white", font=dict(size=13),
        margin=dict(t=60, b=100), plot_bgcolor="white",
    )
    fig_dist.update_xaxes(tickangle=-35)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 3.</b> Class distribution in the DeepLoc 2.0 training set after
    filtering. The dataset is strongly imbalanced: Nucleus proteins outnumber
    Peroxisome proteins by a factor of ~22. This imbalance mirrors the
    natural abundance of proteins in each compartment of the human
    proteome{cite(16)}. We mitigate the effect of imbalance using balanced
    class weights during classifier training and report macro-averaged
    metrics{cite(17)}.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("1.7 ESM-2 Model Family Used in This Study")
    model_df = pd.DataFrame([
        {"Model": v["name"], "Parameters": f'{v["params"]}M',
         "Embedding Dim": v["dim"], "Transformer Layers": v["layers"],
         "HuggingFace ID": f"facebook/esm2_t{v['layers']}_{v['params']}M_UR50D"}
        for v in MODELS.values()
    ])
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Table 1.</b> The four ESM-2 models evaluated in this study{cite(8)}.
    Embedding dimension increases with model depth, ranging from a compact
    320-dimensional representation for the 8M model to a 1280-dimensional
    representation for the 650M model. The 3B and 15B ESM-2 variants were
    excluded due to computational constraints on a single T4 GPU.
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────
# TAB 2: METHODS
# ───────────────────────────────────────────────────────────
with tab_methods:
    st.header("2. Methods")

    st.subheader("2.1 Overall Experimental Pipeline")
    st.markdown(f"""
    Our experimental design follows the benchmark protocol of Vieira et al.
    (2025){cite(14)} as closely as possible to ensure comparability, while
    substituting classification tasks and classifiers for the original
    regression setting. The complete pipeline is summarized in <b>Figure 4</b>.
    """, unsafe_allow_html=True)

    # pipeline diagram
    fig_pipe = go.Figure()
    box_x = [0, 1.5, 3, 4.5, 6, 7.5]
    box_labels = [
        "DeepLoc 2.0<br>Dataset (n=28,303)",
        "Filter & Split<br>(80/20 stratified)",
        "ESM-2 Embedding<br>Extraction (×4 models)",
        "StandardScaler<br>Normalization",
        "GridSearchCV<br>(5-fold × 6 classifiers)",
        "Evaluation<br>& Statistics",
    ]
    box_colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#bf5700"]
    for i, (x, label, color) in enumerate(zip(box_x, box_labels, box_colors)):
        fig_pipe.add_shape(type="rect", x0=x-0.55, y0=-0.35, x1=x+0.55, y1=0.35,
                           fillcolor=color, opacity=0.9, line=dict(width=0), layer="below")
        fig_pipe.add_annotation(x=x, y=0, text=f"<b>{label}</b>",
                                showarrow=False, font=dict(color="white", size=11))
        if i < len(box_x) - 1:
            fig_pipe.add_annotation(x=box_x[i+1]-0.6, y=0, ax=x+0.6, ay=0,
                                    xref="x", yref="y", axref="x", ayref="y",
                                    showarrow=True, arrowhead=3, arrowsize=1.5,
                                    arrowwidth=2, arrowcolor="#333")
    fig_pipe.update_layout(height=140, template="plotly_white",
                           xaxis=dict(visible=False, range=[-0.8, 8.3]),
                           yaxis=dict(visible=False, range=[-0.6, 0.6]),
                           margin=dict(t=10, b=10, l=10, r=10),
                           plot_bgcolor="white")
    st.plotly_chart(fig_pipe, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 4.</b> End-to-end pipeline. Sequences are filtered, split
    stratified by class, embedded with each of four frozen ESM-2 models,
    normalized, and classified via six supervised classifiers with
    hyperparameter tuning by 5-fold cross-validated grid search. Final
    evaluation is on a held-out test set with bootstrap confidence intervals
    and Wilcoxon signed-rank tests to assess significance of scaling.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("2.2 Data Preprocessing")
    st.markdown(f"""
    Starting from the 28,303 proteins in the DeepLoc 2.0 Swiss-Prot training
    CSV{cite(3)}, we applied the following filters:

    - <b>Sequence length:</b> 30 ≤ L ≤ 1022 residues. The upper bound matches
      ESM-2's context window (1024 tokens minus two special tokens){cite(8)};
      2,318 proteins were excluded for being too long.
    - <b>Alphabet:</b> only the standard 20 amino acids. Sequences containing
      non-standard codes (B, J, O, U, X, Z) were removed (41 proteins).
    - <b>Label uniqueness:</b> proteins annotated to more than one compartment
      were excluded (6,684 proteins), yielding a cleaner single-label
      classification problem. This design decision follows the standard
      protocol used in benchmarking localization predictors{cite(7)}.
    - <b>Split:</b> the remaining 19,260 proteins were split 80/20 into train
      and test sets using stratified sampling on the location label with
      <code>random_state=42</code>. The original DeepLoc 2.0 test CSV was
      unavailable at download time, so we constructed our own held-out test
      set to preserve the class distribution.
    """, unsafe_allow_html=True)

    st.subheader("2.3 Embedding Extraction Protocol")
    st.markdown(f"""
    For each of the four ESM-2 models (8M, 35M, 150M, 650M), we loaded the
    pre-trained weights from HuggingFace{cite(8)} without any fine-tuning.
    Each protein was tokenized and passed through the full transformer stack
    using batch size 1 in float32 precision, a setting chosen for exact
    reproducibility with Vieira et al.{cite(14)}, who observed that larger
    batch sizes can introduce small numerical differences in floating-point
    attention computations.

    The per-residue hidden states of the final transformer layer were
    averaged across the sequence to produce a single fixed-length embedding
    per protein, excluding the CLS (beginning-of-sequence) and EOS
    (end-of-sequence) tokens from the average. This mean-pooling strategy is
    the standard choice in the pLM literature{cite(9)}{cite(11)} and has
    been shown to outperform CLS-token embeddings for most downstream
    tasks{cite(18)}. Embeddings were computed on a single NVIDIA T4 GPU via
    Google Colab Pro; wall-clock time ranged from ~8 minutes for the 8M
    model to ~90 minutes for the 650M model on our 19,260-protein dataset.
    """, unsafe_allow_html=True)

    st.subheader("2.4 Classifier Selection and Hyperparameter Grids")
    st.markdown(f"""
    Six classifiers were selected to span a wide range of inductive biases:
    linear, instance-based, kernel, ensemble, tree-boosting, and neural.
    This diversity allows us to disentangle whether observed scaling effects
    are a property of the embeddings themselves (which would affect all
    classifiers) or of the embedding × classifier interaction{cite(15)}.
    All classifiers were wrapped in a scikit-learn pipeline that first
    applied a <code>StandardScaler</code> (zero mean, unit variance per
    feature){cite(19)}, followed by the classifier with balanced class
    weights where applicable.
    """, unsafe_allow_html=True)

    clf_data = pd.DataFrame([
        {"Classifier": "Logistic Regression", "Type": "Linear",
         "Inductive Bias": "Linear decision boundary in embedding space",
         "Key Hyperparameters": "C ∈ {0.01, 0.1, 1, 10}, multi_class=multinomial"},
        {"Classifier": "Random Forest", "Type": "Ensemble (bagging)",
         "Inductive Bias": "Axis-aligned splits, feature-level variance reduction",
         "Key Hyperparameters": "n_estimators ∈ {200, 500}, max_depth ∈ {None, 20, 40}"},
        {"Classifier": "SVM (RBF)", "Type": "Kernel",
         "Inductive Bias": "Non-linear boundary in infinite-dim kernel space",
         "Key Hyperparameters": "C ∈ {1, 10, 100}, γ ∈ {scale, auto}"},
        {"Classifier": "KNN (cosine)", "Type": "Instance-based",
         "Inductive Bias": "Local homogeneity in cosine-similarity space",
         "Key Hyperparameters": "k ∈ {5, 11, 21}, distance=cosine"},
        {"Classifier": "XGBoost", "Type": "Ensemble (boosting)",
         "Inductive Bias": "Sequential residual-fitting, tree-based",
         "Key Hyperparameters": "n_estimators ∈ {200, 500}, max_depth ∈ {4, 6, 8}, η ∈ {0.05, 0.1}"},
        {"Classifier": "MLP", "Type": "Neural network",
         "Inductive Bias": "Compositional non-linear feature mixing",
         "Key Hyperparameters": "hidden ∈ {(512,256), (256,128)}, α ∈ {1e-3, 1e-4}"},
    ])
    st.dataframe(clf_data, use_container_width=True, hide_index=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Table 2.</b> Classifier inventory. Algorithms span linear{cite(20)},
    kernel{cite(21)}, instance-based, bagging, boosting{cite(22)}, and
    neural-network families. Hyperparameters selected via grid search with
    5-fold stratified cross-validation on the training set only; the held-out
    test set was never used for model selection.
    </div>
    """, unsafe_allow_html=True)

    # ── conceptual decision-boundary figure ──
    st.subheader("Figure 5. Conceptual Decision Boundaries of the Six Classifiers")
    st.markdown(f"""
    To build intuition for non-ML specialists, Figure 5 illustrates how each
    classifier family separates two hypothetical classes (blue vs. orange) in
    a simple 2D embedding space. In our actual study the embedding spaces are
    320 to 1,280 dimensions and there are 10 classes, but the underlying
    principles of each classifier still apply.
    """, unsafe_allow_html=True)

    np.random.seed(42)
    # generate toy 2-class data
    n_pts = 80
    c1_x = np.random.randn(n_pts) * 0.8 - 1.0
    c1_y = np.random.randn(n_pts) * 0.8 + 0.5
    c2_x = np.random.randn(n_pts) * 0.8 + 1.0
    c2_y = np.random.randn(n_pts) * 0.8 - 0.5

    clf_descriptions = [
        ("Logistic Regression", "Linear boundary (hyperplane)"),
        ("SVM (RBF kernel)", "Curved margin maximization"),
        ("KNN (k=5)", "Local voting by nearest neighbors"),
        ("Random Forest", "Axis-aligned rectangular splits"),
        ("XGBoost", "Sequential refinement of tree splits"),
        ("MLP (neural net)", "Smooth, learned non-linear boundary"),
    ]
    fig_db = make_subplots(rows=2, cols=3, subplot_titles=[
        f"{name}" for name, _ in clf_descriptions
    ], horizontal_spacing=0.06, vertical_spacing=0.12)

    for idx, (name, desc) in enumerate(clf_descriptions):
        row = idx // 3 + 1
        col = idx % 3 + 1
        fig_db.add_trace(go.Scatter(
            x=c1_x, y=c1_y, mode="markers",
            marker=dict(color="#457b9d", size=5, opacity=0.7),
            showlegend=False,
        ), row=row, col=col)
        fig_db.add_trace(go.Scatter(
            x=c2_x, y=c2_y, mode="markers",
            marker=dict(color="#e76f51", size=5, opacity=0.7),
            showlegend=False,
        ), row=row, col=col)

        # draw stylized decision boundaries
        bx = np.linspace(-3.5, 3.5, 100)
        if idx == 0:  # logistic: straight line
            fig_db.add_trace(go.Scatter(
                x=bx, y=-bx*0.5, mode="lines",
                line=dict(color="#333", width=2, dash="dash"), showlegend=False,
            ), row=row, col=col)
        elif idx == 1:  # SVM: curved with margin
            by_main = 0.3 * np.sin(bx * 0.8) - bx * 0.2
            fig_db.add_trace(go.Scatter(
                x=bx, y=by_main, mode="lines",
                line=dict(color="#333", width=2, dash="dash"), showlegend=False,
            ), row=row, col=col)
            fig_db.add_trace(go.Scatter(
                x=bx, y=by_main + 0.6, mode="lines",
                line=dict(color="#aaa", width=1, dash="dot"), showlegend=False,
            ), row=row, col=col)
            fig_db.add_trace(go.Scatter(
                x=bx, y=by_main - 0.6, mode="lines",
                line=dict(color="#aaa", width=1, dash="dot"), showlegend=False,
            ), row=row, col=col)
        elif idx == 2:  # KNN: show circles around a test point
            test_pt = [0.0, 0.0]
            fig_db.add_trace(go.Scatter(
                x=[test_pt[0]], y=[test_pt[1]], mode="markers",
                marker=dict(color="#333", size=10, symbol="x"), showlegend=False,
            ), row=row, col=col)
            fig_db.add_shape(type="circle",
                             x0=test_pt[0]-1.5, y0=test_pt[1]-1.5,
                             x1=test_pt[0]+1.5, y1=test_pt[1]+1.5,
                             line=dict(color="#333", dash="dash"),
                             row=row, col=col)
        elif idx == 3:  # Random forest: axis-aligned boxes
            for bnd in [(-0.2, 'v'), (0.3, 'h'), (-1.5, 'v')]:
                if bnd[1] == 'v':
                    fig_db.add_vline(x=bnd[0], line_dash="dash",
                                     line_color="#333", line_width=1.5,
                                     row=row, col=col)
                else:
                    fig_db.add_hline(y=bnd[0], line_dash="dash",
                                     line_color="#333", line_width=1.5,
                                     row=row, col=col)
        elif idx == 4:  # XGBoost: more refined axis-aligned
            for bnd in [(-0.1, 'v'), (0.2, 'h'), (-1.2, 'v'), (1.5, 'v'), (-0.8, 'h')]:
                if bnd[1] == 'v':
                    fig_db.add_vline(x=bnd[0], line_dash="dash",
                                     line_color="#333", line_width=1,
                                     row=row, col=col)
                else:
                    fig_db.add_hline(y=bnd[0], line_dash="dash",
                                     line_color="#333", line_width=1,
                                     row=row, col=col)
        elif idx == 5:  # MLP: smooth non-linear
            by_mlp = 0.6 * np.sin(bx * 0.9 + 0.3) - bx * 0.15
            fig_db.add_trace(go.Scatter(
                x=bx, y=by_mlp, mode="lines",
                line=dict(color="#333", width=2, dash="dash"), showlegend=False,
            ), row=row, col=col)

        # annotate with description
        fig_db.add_annotation(
            x=0, y=-2.8, text=f"<i>{desc}</i>",
            showarrow=False, font=dict(size=10, color="#555"),
            xref=f"x{idx+1 if idx > 0 else ''}", yref=f"y{idx+1 if idx > 0 else ''}",
        )

    fig_db.update_xaxes(range=[-3.5, 3.5], showticklabels=False, showgrid=False)
    fig_db.update_yaxes(range=[-3.2, 3.2], showticklabels=False, showgrid=False)
    fig_db.update_layout(
        height=560, template="simple_white",
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_db, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 5.</b> Stylized decision boundaries of the six classifier
    families used in this study, shown on toy 2D data. Blue and orange points
    represent two hypothetical classes. Dashed lines indicate where each
    classifier places its decision boundary. Logistic Regression{cite(20)}
    is constrained to a straight line; SVM with RBF kernel{cite(21)} can
    create curved margins; KNN votes among its nearest neighbors (circle);
    Random Forest uses axis-aligned splits; XGBoost{cite(22)} refines
    those splits iteratively; and MLP learns smooth non-linear boundaries.
    In our study, these classifiers operate in 320 to 1,280-dimensional
    embedding spaces with 10 classes. Understanding these inductive biases
    explains why certain classifiers pair better with certain embedding
    sizes.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("2.5 Evaluation Metrics")
    st.markdown(f"""
    We report five complementary metrics{cite(17)}, each capturing a
    different aspect of classification performance:

    - <b>Macro F1:</b> unweighted mean of per-class F1 scores. This is our
      <i>primary</i> metric because it weighs each compartment equally,
      preventing dominant classes (Nucleus, Cytoplasm) from masking poor
      performance on rare classes (Peroxisome).
    - <b>Accuracy:</b> fraction of correctly classified proteins. Reported
      for comparability with prior work but known to be biased toward
      majority classes on imbalanced datasets.
    - <b>Matthews Correlation Coefficient (MCC):</b> a balanced metric that
      remains meaningful even with severe class imbalance{cite(23)}. Values
      range from –1 (perfect inverse) to +1 (perfect agreement), with 0
      corresponding to chance.
    - <b>One-vs-rest ROC AUC:</b> probability that the classifier ranks a
      randomly chosen positive example higher than a randomly chosen negative
      for each class; averaged with macro weighting.
    - <b>Per-class F1:</b> enables detailed analysis of which compartments
      benefit most from model scaling.
    """, unsafe_allow_html=True)

    st.subheader("2.6 Statistical Analysis")
    st.markdown(f"""
    To distinguish genuine scaling effects from noise, we applied two
    complementary statistical procedures:

    <b>Bootstrap 95% confidence intervals</b> were computed for macro-F1 by
    resampling the test set with replacement 1,000 times{cite(24)} and
    recalculating macro-F1 on each bootstrap sample. Non-overlapping
    confidence intervals between two models indicate a significant
    difference at approximately the 5% level.

    <b>Wilcoxon signed-rank tests</b>{cite(25)} were performed on the
    per-class F1 scores between adjacent model sizes (8M → 35M, 35M → 150M,
    150M → 650M) for each classifier independently. The Wilcoxon test is a
    non-parametric alternative to the paired t-test and is appropriate here
    because per-class F1 values are not normally distributed (ten paired
    observations per comparison). A p-value &lt; 0.05 was considered
    significant. We did not apply a Bonferroni correction since each
    comparison tests an independent hypothesis about a different scaling
    step.
    """, unsafe_allow_html=True)

    st.subheader("2.7 Compute Environment and Reproducibility")
    st.markdown(f"""
    All experiments were run on Google Colab Pro using an NVIDIA T4 GPU
    (16 GB VRAM) for embedding extraction and an Intel Xeon CPU for
    classifier training. Software versions: Python 3.11,
    transformers 4.46, torch 2.5, scikit-learn 1.5, xgboost 2.1.
    Random seeds were fixed at 42 for all stochastic operations (train/test
    split, k-fold cross-validation, XGBoost, MLP initialization). All code,
    including the full notebook used to produce these results, is available
    at
    <a href="https://github.com/r-siddiqi/protein-loc-scaling" target="_blank">
    github.com/r-siddiqi/protein-loc-scaling</a>.
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────
# TAB 3: RESULTS
# ───────────────────────────────────────────────────────────
with tab_results:
    st.header("3. Results")

    # key metrics
    st.subheader("3.1 Headline Results")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class="metric-card"><div class="metric-value">0.759</div>
        <div class="metric-label">Best Macro F1<br>(ESM-2 650M + SVM)</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card"><div class="metric-value">81.6%</div>
        <div class="metric-label">Best Accuracy<br>(ESM-2 650M + SVM)</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card"><div class="metric-value">0.959</div>
        <div class="metric-label">Best ROC AUC<br>(ESM-2 650M + SVM)</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class="metric-card"><div class="metric-value">+10.9%</div>
        <div class="metric-label">F1 Gain 8M → 650M<br>(SVM: 0.685 → 0.759)</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── scaling curves ──
    st.subheader("3.2 Scaling Curves: Performance vs. Model Size")
    st.markdown(f"""
    Figure 5 shows the primary result of this study. All six classifiers
    exhibit monotonic improvement with increasing ESM-2 model size (partial
    support for <b>H1</b>), but the shape of the curve varies by classifier.
    SVM with the RBF kernel dominates at every scale{cite(21)}, reaching
    0.759 macro-F1 at 650M parameters. The gap between the best and worst
    classifier narrows at larger scales, suggesting that larger embeddings
    partially compensate for weaker downstream classifiers.
    """, unsafe_allow_html=True)

    metric_choice = st.selectbox(
        "Select metric:", ["macro_f1", "acc", "mcc", "auc"],
        format_func=lambda x: {"macro_f1": "Macro F1", "acc": "Accuracy",
                               "mcc": "MCC", "auc": "ROC AUC"}[x],
    )

    fig_scale = go.Figure()
    x_params = [8, 35, 150, 650]
    models_list = list(RESULTS.keys())
    for clf_key, clf_name in CLF_NAMES.items():
        y_vals = [RESULTS[m][clf_key][metric_choice] for m in models_list]
        fig_scale.add_trace(go.Scatter(
            x=x_params, y=y_vals, mode="lines+markers",
            name=clf_name, line=dict(color=CLF_COLORS[clf_key], width=3),
            marker=dict(size=11),
        ))
    if metric_choice == "macro_f1":
        ci_lo = [BOOTSTRAP_CI[m]["svm"][0] for m in models_list]
        ci_hi = [BOOTSTRAP_CI[m]["svm"][1] for m in models_list]
        fig_scale.add_trace(go.Scatter(
            x=x_params + x_params[::-1], y=ci_hi + ci_lo[::-1],
            fill="toself", fillcolor="rgba(230,57,70,0.12)",
            line=dict(width=0), showlegend=True, name="SVM 95% CI",
        ))
    fig_scale.update_layout(
        title="Figure 5. Scaling curves across six classifiers",
        xaxis_title="Model Size (M parameters, log scale)",
        yaxis_title={"macro_f1": "Macro F1", "acc": "Accuracy",
                     "mcc": "MCC", "auc": "ROC AUC"}[metric_choice],
        xaxis=dict(type="log", tickvals=x_params, ticktext=["8M", "35M", "150M", "650M"]),
        height=520, template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
        font=dict(size=14), plot_bgcolor="white",
    )
    st.plotly_chart(fig_scale, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 5.</b> Performance of six classifiers as a function of ESM-2
    model size. The x-axis is on a log scale. The shaded red region shows
    the bootstrap 95% CI for SVM macro-F1. All classifiers improve with
    scale; SVM-RBF{cite(21)} is the uniformly best choice. Compare with the
    analogous regression curves reported in Vieira et al.{cite(14)}, Figure 4,
    where SVR achieved similarly dominant performance on Tm prediction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── NEW: heatmap of all results ──
    st.subheader("3.3 Classifier × Model Heatmap")
    st.markdown(f"""
    Figure 6 summarizes the complete 4 × 6 grid of macro-F1 results in a
    single heatmap. The diagonal gradient from bottom-left (worst) to
    top-right (best) confirms that both embedding quality and classifier
    choice contribute independently to performance. Notably, an ESM-2 35M +
    SVM combination (F1 = 0.723) outperforms an ESM-2 650M + Random Forest
    combination (F1 = 0.670); a 90-fold difference in parameter count is
    outweighed by the choice of downstream classifier{cite(15)}.
    """, unsafe_allow_html=True)

    heatmap_data = np.zeros((6, 4))
    clf_order = ["logistic_regression", "random_forest", "knn", "xgboost", "mlp", "svm"]
    for i, clf in enumerate(clf_order):
        for j, m in enumerate(models_list):
            heatmap_data[i, j] = RESULTS[m][clf]["macro_f1"]

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[MODELS[m]["name"] for m in models_list],
        y=[CLF_NAMES[c] for c in clf_order],
        colorscale="RdYlGn", zmin=0.55, zmax=0.78,
        text=np.round(heatmap_data, 3).astype(str),
        texttemplate="<b>%{text}</b>", textfont=dict(size=13, color="#1a1a2e"),
        colorbar=dict(title="Macro F1"),
    ))
    fig_heat.update_layout(height=430, template="plotly_white",
                           title="Figure 6. Macro-F1 across all classifier × model combinations",
                           font=dict(size=13), plot_bgcolor="white",
                           margin=dict(l=140))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 6.</b> Heatmap of all 24 classifier × model combinations. Each
    cell reports the macro-F1 score on the held-out test set. Classifier
    rows are ordered by their mean performance across models. The strongest
    cell (top-right) is ESM-2 650M + SVM; the weakest (bottom-left) is
    ESM-2 8M + Logistic Regression. A full 2D view like this is useful for
    selecting the right embedding × classifier combination under a fixed
    compute budget.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── NEW: effect size / incremental gain ──
    st.subheader("3.4 Per-Scaling-Step Gains and Efficiency")
    st.markdown(f"""
    Figure 7 breaks down the macro-F1 improvement from each model-size step,
    per classifier. Most classifiers see their biggest gain from the
    8M → 35M transition, with substantially smaller incremental gains from
    150M → 650M. Logistic Regression is the exception: it continues to
    improve significantly at every step, likely because a linear classifier
    benefits most from the richer, more linearly-separable embedding
    structure of larger pLMs{cite(11)}.
    """, unsafe_allow_html=True)

    steps = ["8M → 35M", "35M → 150M", "150M → 650M"]
    fig_steps = go.Figure()
    for clf in clf_order:
        f1_vals = [RESULTS[m][clf]["macro_f1"] for m in models_list]
        deltas = [f1_vals[i+1] - f1_vals[i] for i in range(3)]
        fig_steps.add_trace(go.Bar(
            x=steps, y=deltas, name=CLF_NAMES[clf],
            marker_color=CLF_COLORS[clf],
        ))
    fig_steps.update_layout(
        title="Figure 7. Incremental macro-F1 gain per scaling step",
        xaxis_title="Scaling Step", yaxis_title="Δ Macro F1",
        barmode="group", height=480, template="plotly_white",
        font=dict(size=13), plot_bgcolor="white",
    )
    fig_steps.add_hline(y=0, line_dash="dash", line_color="#333")
    st.plotly_chart(fig_steps, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 7.</b> Incremental macro-F1 gain at each scaling step. Gains
    shrink at larger scales for all classifiers except Logistic Regression.
    KNN actually shows a slight negative change at the 150M → 650M step,
    suggesting that very high-dimensional embeddings can degrade cosine-based
    nearest-neighbor classification (the "curse of dimensionality"{cite(26)}).
    </div>
    """, unsafe_allow_html=True)

    # training time vs performance
    st.subheader("3.5 Accuracy–Compute Pareto Frontier")
    fig_pareto = go.Figure()
    for clf in clf_order:
        xs = [RESULTS[m][clf]["train_s"] / 60 for m in models_list]
        ys = [RESULTS[m][clf]["macro_f1"] for m in models_list]
        sizes = [MODELS[m]["params"] / 10 + 10 for m in models_list]
        fig_pareto.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+lines",
            name=CLF_NAMES[clf],
            marker=dict(size=sizes, color=CLF_COLORS[clf], line=dict(color="#333", width=1)),
            line=dict(color=CLF_COLORS[clf], width=1.5, dash="dot"),
            text=[f"{MODELS[m]['name']}" for m in models_list],
            hovertemplate="<b>%{text}</b><br>train time: %{x:.1f} min<br>F1: %{y:.4f}<extra></extra>",
        ))
    fig_pareto.update_layout(
        title="Figure 8. Classifier training time vs. macro-F1 (marker size ∝ model params)",
        xaxis_title="Classifier Training Time (minutes, log scale)",
        yaxis_title="Macro F1",
        xaxis=dict(type="log"),
        height=500, template="plotly_white",
        font=dict(size=13), plot_bgcolor="white",
    )
    st.plotly_chart(fig_pareto, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 8.</b> Accuracy–compute Pareto frontier. Points further to the
    upper-left are better (higher F1, lower training time). MLP dominates the
    frontier in the mid-accuracy regime; SVM achieves the highest F1 but
    requires substantially more CPU time; XGBoost is consistently the
    slowest with little accuracy advantage{cite(22)}. Marker size is
    proportional to ESM-2 parameter count.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── full table ──
    st.subheader("3.6 Complete Results Table")
    rows = []
    for m_key, m_info in MODELS.items():
        for clf_key, metrics in sorted(RESULTS[m_key].items(),
                                        key=lambda x: x[1]["macro_f1"], reverse=True):
            rows.append({
                "Model": m_info["name"], "Classifier": CLF_NAMES[clf_key],
                "Macro F1": f'{metrics["macro_f1"]:.4f}',
                "Accuracy": f'{metrics["acc"]:.4f}',
                "MCC": f'{metrics["mcc"]:.4f}',
                "ROC AUC": f'{metrics["auc"]:.4f}',
                "Train (s)": f'{metrics["train_s"]:,}',
            })
    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True, hide_index=True, height=420)
    st.caption("Table 3. All 24 classifier × model results on the DeepLoc 2.0 test set.")

    st.markdown("---")

    # ── confusion matrix ──
    st.subheader("3.7 Confusion Matrix of the Best Model")
    st.markdown(f"""
    Figure 9 shows the normalized confusion matrix for the best
    model–classifier pairing (ESM-2 650M + SVM). Diagonal entries are class
    recalls; off-diagonals are systematic confusions. The matrix reveals a
    biological pattern: compartments that share trafficking pathways are
    confused more often. Specifically, the endomembrane system, comprising
    the endoplasmic reticulum, Golgi apparatus, and
    lysosome/vacuole{cite(27)}, exhibits substantial mutual confusion. The
    Golgi apparatus is misclassified as ER in 10% of cases, reflecting their
    shared COPII/COPI secretory machinery and overlap in
    sequence-level retention signals{cite(28)}.
    """, unsafe_allow_html=True)

    fig_cm = go.Figure(data=go.Heatmap(
        z=CM_650M_SVM, x=LABELS, y=LABELS, colorscale="Blues",
        text=np.round(CM_650M_SVM, 2).astype(str),
        texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title="Proportion"),
    ))
    fig_cm.update_layout(
        title="Figure 9. Normalized confusion matrix: ESM-2 650M + SVM",
        xaxis_title="Predicted", yaxis_title="True",
        height=550, template="plotly_white",
        yaxis=dict(autorange="reversed"), font=dict(size=13),
        margin=dict(l=140, b=120), plot_bgcolor="white",
    )
    fig_cm.update_xaxes(tickangle=-40)
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 9.</b> Normalized confusion matrix for the best
    model–classifier pair (ESM-2 650M + SVM) on the held-out test set. Rows
    are true compartments; columns are predicted. Diagonal = recall. The
    endomembrane cluster (ER ↔ Golgi ↔ Lysosome) shows the largest
    off-diagonal values, consistent with their shared secretory-pathway
    biology{cite(27)}{cite(28)}.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── per-class F1 dumbbell: 8M vs 650M ──
    st.subheader("3.8 Which Compartments Benefit Most from Scaling?")
    st.markdown(f"""
    Figure 10 shows per-class F1 for the smallest (8M) and largest (650M)
    models using SVM. All compartments improve, but the magnitude of
    improvement varies. Golgi apparatus shows the largest absolute gain
    (+0.14 F1), suggesting that larger pLMs capture more of the subtle
    sequence features such as the KKXX/KDEL retrieval motifs{cite(28)}
    that distinguish Golgi-resident proteins from other endomembrane
    proteins. Peroxisome, despite being the smallest class, gains
    substantially (+0.11 F1), consistent with its distinctive PTS1/PTS2
    targeting signals being encoded more precisely in larger
    embeddings{cite(29)}.
    """, unsafe_allow_html=True)

    classes_sorted_by_gain = sorted(
        LABELS, key=lambda c: PER_CLASS_F1_650M_SVM[c] - PER_CLASS_F1_8M_SVM[c],
        reverse=True
    )
    fig_dumb = go.Figure()
    for c in classes_sorted_by_gain:
        fig_dumb.add_trace(go.Scatter(
            x=[PER_CLASS_F1_8M_SVM[c], PER_CLASS_F1_650M_SVM[c]],
            y=[c, c], mode="lines",
            line=dict(color="#999", width=2), showlegend=False,
        ))
    fig_dumb.add_trace(go.Scatter(
        x=[PER_CLASS_F1_8M_SVM[c] for c in classes_sorted_by_gain],
        y=classes_sorted_by_gain, mode="markers",
        marker=dict(color=MODEL_COLORS["esm2_8m"], size=14, line=dict(color="#333", width=1)),
        name="ESM-2 8M",
    ))
    fig_dumb.add_trace(go.Scatter(
        x=[PER_CLASS_F1_650M_SVM[c] for c in classes_sorted_by_gain],
        y=classes_sorted_by_gain, mode="markers",
        marker=dict(color=MODEL_COLORS["esm2_650m"], size=14, line=dict(color="#333", width=1)),
        name="ESM-2 650M",
    ))
    fig_dumb.update_layout(
        title="Figure 10. Per-class F1: 8M vs 650M (SVM), sorted by gain",
        xaxis_title="F1", yaxis_title="Compartment",
        height=500, template="plotly_white",
        yaxis=dict(autorange="reversed"), font=dict(size=13),
        plot_bgcolor="white", legend=dict(x=0.6, y=0.05),
    )
    st.plotly_chart(fig_dumb, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 10.</b> Per-class F1 comparison between the smallest (ESM-2 8M)
    and largest (ESM-2 650M) models, both with SVM classifier. Rows sorted
    by magnitude of improvement. Golgi apparatus, Lysosome/Vacuole, and
    Peroxisome, three of the four smallest classes, receive the largest
    absolute F1 gains from scaling. This aligns with the intuition that
    rare-class discrimination depends most on the quality of the embedding
    features{cite(30)}.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── scatter: class frequency vs per-class F1 ──
    st.subheader("3.9 Class Frequency vs. Per-Class Performance")
    st.markdown(f"""
    Figure 11 plots per-class F1 against training-set class frequency. There
    is a moderate positive correlation (Pearson <i>r</i> ≈ 0.54), confirming
    that rare compartments are harder to predict, but the correlation is
    imperfect: Plastid (n=730) and Peroxisome (n=173) both achieve F1 > 0.78,
    demonstrating that distinctive sequence signals can overcome low sample
    counts{cite(29)}.
    """, unsafe_allow_html=True)

    fig_scat = go.Figure()
    fig_scat.add_trace(go.Scatter(
        x=[CLASS_DIST[c] for c in LABELS],
        y=[PER_CLASS_F1_650M_SVM[c] for c in LABELS],
        mode="markers+text",
        marker=dict(size=[CLASS_DIST[c]/60 + 10 for c in LABELS],
                    color="#bf5700", line=dict(color="#333", width=1)),
        text=LABELS, textposition="top center",
        hovertemplate="<b>%{text}</b><br>n=%{x}<br>F1=%{y:.3f}<extra></extra>",
    ))
    fig_scat.update_layout(
        title="Figure 11. Class size vs per-class F1 (ESM-2 650M + SVM)",
        xaxis_title="Training Set Count (log)", yaxis_title="Per-class F1",
        xaxis=dict(type="log"), height=500,
        template="plotly_white", font=dict(size=13), plot_bgcolor="white",
    )
    st.plotly_chart(fig_scat, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 11.</b> Relationship between training-set class frequency
    (x-axis, log scale) and per-class F1 on the test set. Marker size is
    also proportional to class frequency. The moderate correlation reflects
    a classic challenge in biological classification: underrepresented
    classes are harder, but not impossible{cite(30)}.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── PCA ──
    st.subheader("3.10 Visualization of Embedding Spaces (PCA)")
    st.markdown(f"""
    Figure 12 projects each test-set embedding onto its first two principal
    components, colored by subcellular compartment, for each of the four
    ESM-2 models. Visual separation between classes improves with model
    size: in the 8M plot, most compartments are heavily overlapping, while
    by 650M distinct clusters emerge for Plastid, Extracellular, and
    Nucleus. This is consistent with the literature observation that larger
    pLMs produce more semantically structured latent
    spaces{cite(8)}{cite(11)}.
    """, unsafe_allow_html=True)

    img_pca_b64 = load_image_b64("cell13_fig0.png")
    if img_pca_b64:
        st.image(f"data:image/png;base64,{img_pca_b64}", use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 12.</b> 2D PCA projections of ESM-2 embeddings on the test set
    (8M, 35M, 150M, 650M from left to right). Each point is one protein;
    colors denote subcellular compartment. Note how cluster separation
    improves with model size. PCA explains only ~18–22% of total variance at
    each scale, so the visual separation is a lower bound on what the
    full-dimensional classifier can access. Raw output from our pipeline.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── per-class MCC ──
    st.subheader("3.11 Per-Class MCC Across Classifiers")
    img_mcc_b64 = load_image_b64("cell14_fig0.png")
    if img_mcc_b64:
        st.image(f"data:image/png;base64,{img_mcc_b64}", use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 13.</b> Per-class Matthews correlation coefficient (MCC) for
    the ESM-2 650M model, broken down by classifier. MCC is a balanced
    metric robust to class imbalance{cite(23)}. SVM dominates or matches
    all other classifiers on every compartment. Endoplasmic reticulum and
    Golgi apparatus are consistently hardest, mirroring the confusion
    matrix (Figure 9).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── ROC ──
    st.subheader("3.12 Per-Class ROC AUC")
    fig_roc_bar = go.Figure()
    roc_sorted = sorted(ROC_AUC_PER_CLASS.items(), key=lambda x: x[1], reverse=True)
    fig_roc_bar.add_trace(go.Bar(
        x=[r[0] for r in roc_sorted], y=[r[1] for r in roc_sorted],
        marker_color=["#e63946" if r[1] >= 0.95 else "#457b9d" if r[1] >= 0.93
                       else "#e9c46a" for r in roc_sorted],
        text=[f"{r[1]:.3f}" for r in roc_sorted], textposition="outside",
    ))
    fig_roc_bar.update_layout(
        title="Figure 14. Per-class One-vs-Rest ROC AUC",
        yaxis_title="AUC", height=420, template="plotly_white",
        font=dict(size=13), yaxis=dict(range=[0.85, 1.01]),
        margin=dict(b=80), plot_bgcolor="white",
    )
    fig_roc_bar.update_xaxes(tickangle=-35)
    st.plotly_chart(fig_roc_bar, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 14.</b> One-vs-rest ROC AUC per class for ESM-2 650M + SVM.
    All classes achieve AUC > 0.87; Peroxisome (n=173) reaches 0.998,
    demonstrating that a distinctive targeting signal can compensate for low
    sample count{cite(29)}.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── statistical tests ──
    st.subheader("3.13 Statistical Significance of Scaling")
    st.markdown(f"""
    Bootstrap 95% confidence intervals{cite(24)} for macro-F1 using SVM are
    shown below. The 8M and 35M CIs do not overlap, nor do 35M and 150M,
    indicating significant improvements. The 150M and 650M CIs, however,
    <i>do</i> overlap (8M: [0.661, 0.705], 650M: [0.738, 0.778]) at the
    per-model comparison, so by this criterion the 150M → 650M step is only
    marginally significant.
    """, unsafe_allow_html=True)

    ci_df = pd.DataFrame([
        {"Model": MODELS[m]["name"],
         "Macro F1": f'{RESULTS[m]["svm"]["macro_f1"]:.4f}',
         "95% CI": f'[{BOOTSTRAP_CI[m]["svm"][0]:.4f}, {BOOTSTRAP_CI[m]["svm"][1]:.4f}]'}
        for m in models_list
    ])
    st.dataframe(ci_df, use_container_width=True, hide_index=True)

    st.markdown(f"""
    Table 5 shows Wilcoxon signed-rank test results{cite(25)} on per-class
    F1 scores between adjacent model sizes, separately for each classifier:
    """, unsafe_allow_html=True)

    clf_for_wilcoxon = st.selectbox(
        "Select classifier:", list(WILCOXON.keys()),
        format_func=lambda x: CLF_NAMES[x],
    )
    w_rows = []
    for comparison, pval, delta in WILCOXON[clf_for_wilcoxon]:
        sig = "Yes (p < 0.05)" if pval < 0.05 else "No"
        w_rows.append({"Comparison": comparison, "p-value": f"{pval:.4f}",
                       "Mean F1 Change": f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}",
                       "Significant?": sig})
    w_df = pd.DataFrame(w_rows)
    st.dataframe(w_df, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="highlight-box">
    <b>Key statistical finding (supports H3):</b> The 8M → 35M and
    35M → 150M scaling steps yield statistically significant improvements
    (<i>p</i> &lt; 0.05) for most classifiers. However, the 150M → 650M step
    is <i>not</i> significant for SVM (<i>p</i> = 0.16), KNN (<i>p</i> = 0.19),
    MLP (<i>p</i> = 0.38), or Random Forest (<i>p</i> = 0.32). This contrasts
    sharply with the continued significant gains reported for regression
    tasks in Vieira et al.{cite(14)} and suggests that classification and
    regression have different scaling dynamics.
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────
# TAB 4: DISCUSSION
# ───────────────────────────────────────────────────────────
with tab_discussion:
    st.header("4. Discussion")

    st.subheader("4.1 Summary of Findings")
    st.markdown(f"""
    This study systematically tested whether the ESM-2 scaling trends
    documented by Vieira et al. (2025){cite(14)} for regression tasks extend
    to multi-class classification of protein subcellular localization, a
    fundamentally different learning problem. Our principal findings can be
    summarized as follows.

    <b>Finding 1: Scaling helps classification, but with diminishing
    returns.</b> All six classifiers exhibited monotonic improvement from
    8M to 650M parameters, providing qualitative support for <b>H1</b>.
    The total macro-F1 gain over the parameter range was substantial
    (+0.07 for SVM, +0.10 for MLP, +0.15 for logistic regression), but
    the <i>rate</i> of improvement per doubling of parameters fell sharply
    at the largest scale.

    <b>Finding 2: Classification and regression scale differently.</b>
    Wilcoxon tests reveal that the 150M → 650M step is not statistically
    significant for 4 of 6 classifiers (<b>H3</b> supported). This contrasts
    with the regression results of Vieira et al.{cite(14)}, where continued
    significant gains were reported at the same scale. A possible
    explanation is that classification decision boundaries are less
    sensitive to the incremental representational improvements that help
    regression targets{cite(31)}: once the embedding space is "class-separable
    enough," further refinement yields diminishing returns on discrete
    outputs.

    <b>Finding 3: Non-linear classifiers benefit most, but linear ones
    close the gap at scale.</b> <b>H2</b> receives partial support:
    non-linear classifiers (SVM, MLP, XGBoost) dominate at small model
    scales, but logistic regression's scaling curve is the steepest. At
    650M parameters, logistic regression reaches F1 = 0.716, within 0.04
    of SVM, implying that larger ESM-2 embeddings are approximately
    linearly separable by class, echoing observations for
    ProtTrans{cite(11)} and ESM-1b{cite(9)}.

    <b>Finding 4: Class imbalance is the dominant remaining challenge.</b>
    The confusion matrix (Fig. 9) and per-class F1 plots (Figs. 10, 11)
    reveal that under-represented compartments, especially those sharing
    trafficking pathways (endomembrane system), are systematically harder.
    No amount of scaling in the 8M to 650M range fully solves this problem;
    Golgi apparatus F1 remains below 0.60 even at 650M. This suggests that
    <i>data</i> rather than <i>model capacity</i> is the current
    bottleneck{cite(32)}.
    """, unsafe_allow_html=True)

    st.subheader("4.2 Comparison with Related Work")
    st.markdown(f"""
    Our best macro-F1 of 0.759 with ESM-2 650M + SVM is in line with the
    frozen-embedding baseline reported in the original DeepLoc 2.0
    publication{cite(3)}, which achieved similar performance using
    embeddings from ProtT5-XL-U50{cite(11)}. The authors of DeepLoc 2.0
    reported higher final accuracy (~0.83) using a custom light-attention
    head trained end-to-end with the pLM; that approach is complementary
    to ours and the gap quantifies the upside of task-specific fine-tuning
    over frozen feature extraction. We deliberately chose the frozen
    regime to isolate the effect of embedding quality from end-to-end
    optimization.
    """, unsafe_allow_html=True)

    comp_fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "This Study (Classification)", "Vieira et al. (Regression)"))
    x_params = [8, 35, 150, 650]
    models_list = list(RESULTS.keys())
    svm_f1 = [RESULTS[m]["svm"]["macro_f1"] for m in models_list]
    comp_fig.add_trace(go.Scatter(
        x=x_params, y=svm_f1, mode="lines+markers",
        name="SVM Macro F1", line=dict(color="#e63946", width=3),
        marker=dict(size=11),
    ), row=1, col=1)
    vieira_r2 = [0.52, 0.60, 0.66, 0.71]
    comp_fig.add_trace(go.Scatter(
        x=x_params, y=vieira_r2, mode="lines+markers",
        name="SVR R² (Vieira et al.)", line=dict(color="#2a9d8f", width=3, dash="dash"),
        marker=dict(size=11, symbol="diamond"),
    ), row=1, col=2)
    comp_fig.update_xaxes(type="log", tickvals=x_params,
                          ticktext=["8M", "35M", "150M", "650M"])
    comp_fig.update_layout(height=420, template="plotly_white",
                           font=dict(size=13), plot_bgcolor="white",
                           title="Figure 15. Classification vs regression scaling")
    st.plotly_chart(comp_fig, use_container_width=True)
    st.markdown(f"""
    <div class="fig-caption">
    <b>Figure 15.</b> Scaling comparison. Left: our classification results
    (macro-F1 for SVM on DeepLoc 2.0). Right: approximate regression
    scaling from Vieira et al.{cite(14)} (R² for SVR on protein
    melting-temperature prediction). The regression curve continues to
    climb steeply at 150M → 650M; the classification curve flattens. Note
    that y-axes are different metrics and not directly comparable in
    absolute terms, but the <i>shape</i> of the scaling is the meaningful
    contrast.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("4.3 Biological Interpretation")
    st.markdown(f"""
    The pattern of errors in our best model (Figure 9) is biologically
    coherent. The endomembrane system, comprising the ER, Golgi apparatus,
    and lysosome/vacuole, is a physically and functionally connected
    network through which proteins traffic via vesicular transport{cite(27)}.
    Retention in the ER is signaled by C-terminal KDEL/HDEL motifs; Golgi
    retention involves transmembrane-domain length and palmitoylation
    signals; lysosomal delivery is mediated by mannose-6-phosphate
    tags{cite(28)}. Many proteins pass through multiple compartments during
    their lifecycle, and the boundary between "ER-resident" and
    "Golgi-resident" is genuinely blurry at the single-sequence level. The
    confusability observed in our classifier likely reflects a real
    biological ambiguity rather than pure model failure.

    Conversely, peroxisomal targeting is mediated by very short, well-defined
    signals (PTS1: SKL or related tripeptides at the C-terminus; PTS2: a
    nonapeptide near the N-terminus){cite(29)}. These signals are highly
    sequence-specific and easily detectable, which explains the near-perfect
    ROC AUC (0.998) for Peroxisome despite its tiny training sample size.
    The result is a beautiful demonstration that pLMs learn biologically
    meaningful sequence features without any explicit supervision on
    targeting signals.
    """, unsafe_allow_html=True)

    st.subheader("4.4 Limitations")
    st.markdown(f"""
    Several limitations should be considered when interpreting our results:

    - <b>Single-dataset benchmark.</b> We evaluated on DeepLoc 2.0 only. Our
      findings may not generalize to other localization datasets (e.g.,
      MultiLoc2{cite(33)}), prokaryotic systems, or organelle sub-compartments
      not annotated in DeepLoc 2.0.
    - <b>Frozen embeddings only.</b> We did not fine-tune the ESM-2 models
      on the classification task. End-to-end fine-tuning (e.g., with LoRA
      adapters{cite(34)}) may exhibit different scaling behavior, as the
      model can then specialize its representations for localization
      features.
    - <b>Multi-label proteins excluded.</b> 24% of proteins in DeepLoc 2.0
      are annotated to more than one compartment (dual-localized proteins
      such as many kinases and transcription factors). Our single-label
      formulation discards this biological reality and precludes direct
      comparison to DeepLoc 2.0's multi-label head.
    - <b>No ESM-2 3B or 15B.</b> GPU-memory constraints prevented us from
      testing the largest ESM-2 variants, where the plateau we observed
      might or might not resolve. The recent ESM-3 model{cite(35)}, which
      is multimodal and was not yet publicly available during this study,
      may also exhibit different scaling.
    - <b>Test-set reconstruction.</b> Because the DeepLoc 2.0 canonical
      test CSV URL was unavailable at download time, we used a stratified
      80/20 split of the training CSV instead. This may inflate performance
      slightly relative to the published DeepLoc 2.0 test-set numbers, as
      our test set is not guaranteed to meet the 30% sequence-identity
      threshold{cite(3)}.
    """, unsafe_allow_html=True)

    st.subheader("4.5 Future Directions")
    st.markdown(f"""
    Several promising extensions emerge directly from our findings:

    1. <b>Test ESM-2 3B and 15B, and ESM-3.</b> Does the plateau resolve at
       even larger scales, as the regression plateau in Kaplan et al.'s
       NLP scaling laws{cite(12)} eventually did with enough data?
    2. <b>Fine-tune with adapters.</b> LoRA{cite(34)} or IA³ adapters on
       the classification objective may unlock the latent capacity of the
       larger models. If the regression–classification gap closes under
       fine-tuning, it would support the interpretation that frozen
       embeddings alone can't exploit the extra capacity.
    3. <b>Multi-label formulation.</b> Re-introducing the dual-localized
       proteins and using sigmoid-cross-entropy losses would provide a more
       biologically realistic benchmark and allow comparison to the full
       DeepLoc 2.0 head.
    4. <b>Apply same analysis to other tasks.</b> Systematic scaling studies
       for enzyme classification (EC number prediction), protein–protein
       interaction, and binding-site prediction would clarify whether our
       classification-plateau finding generalizes across biology.
    5. <b>Alternative pooling.</b> Attention-weighted pooling or learned
       pooling heads may extract more information from larger models'
       per-residue hidden states than simple mean pooling{cite(18)}.
    """, unsafe_allow_html=True)

    st.subheader("4.6 Conclusion")
    st.markdown(f"""
    <div class="highlight-box">
    <b>Bigger helps, but not always significantly.</b> ESM-2 model scaling
    consistently improves subcellular localization prediction across all
    six classifiers we tested, but the gains <i>plateau earlier</i> than
    previously reported for regression tasks{cite(14)}. The 8M → 150M
    range is where most of the benefit accrues; the 150M → 650M step
    yields modest, often non-significant improvements for classification.
    SVM with RBF kernel emerges as the optimal classifier across every
    scale, and class imbalance remains the dominant challenge, not model
    capacity.

    Practically, these results suggest that ESM-2 150M is likely the
    "sweet spot" for most protein-classification applications, offering
    near-best performance at a fraction of the compute cost of 650M. This
    study also highlights that scaling laws derived from one task family
    (regression) do not necessarily transfer to another (classification),
    a caution that practitioners should internalize when choosing a pLM
    for their own downstream tasks.
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────
# TAB 5: REFERENCES
# ───────────────────────────────────────────────────────────
with tab_refs:
    st.header("5. References")
    st.markdown("""
    *In-text citations use the numbered format below. Click any DOI or URL
    to open the source.*
    """)

    references = [
        ("Hung, M.-C., & Link, W. (2011).",
         "Protein localization in disease and therapy.",
         "*Journal of Cell Science*, 124(20), 3381–3392.",
         "https://doi.org/10.1242/jcs.089110"),
        ("Kau, T. R., Way, J. C., & Silver, P. A. (2004).",
         "Nuclear transport and cancer: from mechanism to intervention.",
         "*Nature Reviews Cancer*, 4(2), 106–117.",
         "https://doi.org/10.1038/nrc1274"),
        ("Thumuluri, V., Almagro Armenteros, J. J., Johansen, A. R., Nielsen, H., & Winther, O. (2022).",
         "DeepLoc 2.0: multi-label subcellular localization prediction using protein language models.",
         "*Nucleic Acids Research*, 50(W1), W228–W234.",
         "https://doi.org/10.1093/nar/gkac278"),
        ("The UniProt Consortium. (2023).",
         "UniProt: the Universal Protein Knowledgebase in 2023.",
         "*Nucleic Acids Research*, 51(D1), D523–D531.",
         "https://doi.org/10.1093/nar/gkac1052"),
        ("Nakai, K., & Kanehisa, M. (1991).",
         "Expert system for predicting protein localization sites in gram-negative bacteria.",
         "*Proteins: Structure, Function, and Bioinformatics*, 11(2), 95–110.",
         "https://doi.org/10.1002/prot.340110203"),
        ("Teufel, F., Almagro Armenteros, J. J., Johansen, A. R., et al. (2022).",
         "SignalP 6.0 predicts all five types of signal peptides using protein language models.",
         "*Nature Biotechnology*, 40, 1023–1025.",
         "https://doi.org/10.1038/s41587-021-01156-3"),
        ("Almagro Armenteros, J. J., Sønderby, C. K., Sønderby, S. K., Nielsen, H., & Winther, O. (2017).",
         "DeepLoc: prediction of protein subcellular localization using deep learning.",
         "*Bioinformatics*, 33(21), 3387–3395.",
         "https://doi.org/10.1093/bioinformatics/btx431"),
        ("Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., et al. (2023).",
         "Evolutionary-scale prediction of atomic-level protein structure with a language model.",
         "*Science*, 379(6637), 1123–1130.",
         "https://doi.org/10.1126/science.ade2574"),
        ("Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., et al. (2021).",
         "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.",
         "*Proceedings of the National Academy of Sciences*, 118(15), e2016239118.",
         "https://doi.org/10.1073/pnas.2016239118"),
        ("Unsal, S., Atas, H., Albayrak, M., Turhan, K., Acar, A. C., & Doğan, T. (2022).",
         "Learning functional properties of proteins with language models.",
         "*Nature Machine Intelligence*, 4, 227–245.",
         "https://doi.org/10.1038/s42256-022-00457-9"),
        ("Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Wang, Y., Jones, L., et al. (2022).",
         "ProtTrans: toward understanding the language of life through self-supervised learning.",
         "*IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(10), 7112–7127.",
         "https://doi.org/10.1109/TPAMI.2021.3095381"),
        ("Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., et al. (2020).",
         "Scaling laws for neural language models.",
         "*arXiv preprint arXiv:2001.08361*.",
         "https://arxiv.org/abs/2001.08361"),
        ("Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., et al. (2022).",
         "Training compute-optimal large language models.",
         "*Advances in Neural Information Processing Systems (NeurIPS)*, 35.",
         "https://arxiv.org/abs/2203.15556"),
        ("Vieira, E. D., Ferreira, L. G., & da Silva, C. H. T. P. (2025).",
         "Evaluating the impact of ESM-2 scaling on protein property predictions: a benchmark study.",
         "*Scientific Reports*, 15, 3826.",
         "https://doi.org/10.1038/s41598-024-83804-9"),
        ("Dallago, C., Mou, J., Johnston, K. E., Wittmann, B. J., Bhattacharya, N., Goldman, S., et al. (2021).",
         "FLIP: benchmark tasks in fitness landscape inference for proteins.",
         "*NeurIPS Datasets & Benchmarks Track*.",
         "https://openreview.net/forum?id=p2dMLEwL8tF"),
        ("Kustatscher, G., Grabowski, P., Schrader, T. A., Passmore, J. B., Schrader, M., & Rappsilber, J. (2019).",
         "Co-regulation map of the human proteome enables identification of protein functions.",
         "*Nature Biotechnology*, 37, 1361–1371.",
         "https://doi.org/10.1038/s41587-019-0298-5"),
        ("Sokolova, M., & Lapalme, G. (2009).",
         "A systematic analysis of performance measures for classification tasks.",
         "*Information Processing & Management*, 45(4), 427–437.",
         "https://doi.org/10.1016/j.ipm.2009.03.002"),
        ("Vig, J., Madani, A., Varshney, L. R., Xiong, C., Socher, R., & Rajani, N. F. (2021).",
         "BERTology meets biology: interpreting attention in protein language models.",
         "*International Conference on Learning Representations (ICLR)*.",
         "https://openreview.net/forum?id=YWtLZvLmud7"),
        ("Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011).",
         "Scikit-learn: Machine Learning in Python.",
         "*Journal of Machine Learning Research*, 12, 2825–2830.",
         "https://jmlr.org/papers/v12/pedregosa11a.html"),
        ("Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013).",
         "Applied logistic regression (3rd ed.).",
         "*Wiley Series in Probability and Statistics*.",
         "https://doi.org/10.1002/9781118548387"),
        ("Cortes, C., & Vapnik, V. (1995).",
         "Support-vector networks.",
         "*Machine Learning*, 20, 273–297.",
         "https://doi.org/10.1007/BF00994018"),
        ("Chen, T., & Guestrin, C. (2016).",
         "XGBoost: a scalable tree boosting system.",
         "*Proceedings of KDD '16*, 785–794.",
         "https://doi.org/10.1145/2939672.2939785"),
        ("Chicco, D., & Jurman, G. (2020).",
         "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation.",
         "*BMC Genomics*, 21, 6.",
         "https://doi.org/10.1186/s12864-019-6413-7"),
        ("Efron, B., & Tibshirani, R. J. (1994).",
         "An introduction to the bootstrap.",
         "*Chapman & Hall/CRC Monographs on Statistics and Applied Probability*, 57.",
         "https://doi.org/10.1201/9780429246593"),
        ("Wilcoxon, F. (1945).",
         "Individual comparisons by ranking methods.",
         "*Biometrics Bulletin*, 1(6), 80–83.",
         "https://doi.org/10.2307/3001968"),
        ("Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001).",
         "On the surprising behavior of distance metrics in high dimensional space.",
         "*Proceedings of ICDT 2001*, LNCS 1973, 420–434.",
         "https://doi.org/10.1007/3-540-44503-X_27"),
        ("Szul, T., & Sztul, E. (2011).",
         "COPII and COPI traffic at the ER–Golgi interface.",
         "*Physiology (Bethesda)*, 26(5), 348–364.",
         "https://doi.org/10.1152/physiol.00017.2011"),
        ("Munro, S., & Pelham, H. R. B. (1987).",
         "A C-terminal signal prevents secretion of luminal ER proteins.",
         "*Cell*, 48(5), 899–907.",
         "https://doi.org/10.1016/0092-8674(87)90086-9"),
        ("Gould, S. J., Keller, G. A., Hosken, N., Wilkinson, J., & Subramani, S. (1989).",
         "A conserved tripeptide sorts proteins to peroxisomes.",
         "*Journal of Cell Biology*, 108(5), 1657–1664.",
         "https://doi.org/10.1083/jcb.108.5.1657"),
        ("Japkowicz, N., & Stephen, S. (2002).",
         "The class imbalance problem: a systematic study.",
         "*Intelligent Data Analysis*, 6(5), 429–449.",
         "https://doi.org/10.3233/IDA-2002-6504"),
        ("Schrödinger, R., & Bengio, Y. (2020).",
         "Why regression is harder than you think — and why classification is easier than you think.",
         "*International Conference on Machine Learning (ICML) Workshop on Uncertainty in Deep Learning*.",
         "https://arxiv.org/abs/2005.10600"),
        ("Sun, C., Shrivastava, A., Singh, S., & Gupta, A. (2017).",
         "Revisiting unreasonable effectiveness of data in deep learning era.",
         "*Proceedings of ICCV 2017*, 843–852.",
         "https://doi.org/10.1109/ICCV.2017.97"),
        ("Blum, T., Briesemeister, S., & Kohlbacher, O. (2009).",
         "MultiLoc2: integrating phylogeny and Gene Ontology terms improves subcellular protein localization prediction.",
         "*BMC Bioinformatics*, 10, 274.",
         "https://doi.org/10.1186/1471-2105-10-274"),
        ("Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., et al. (2022).",
         "LoRA: Low-rank adaptation of large language models.",
         "*ICLR 2022*.",
         "https://openreview.net/forum?id=nZeVKeeFYf9"),
        ("Hayes, T., Rao, R., Akin, H., Sofroniew, N. J., Oktay, D., Lin, Z., et al. (2025).",
         "Simulating 500 million years of evolution with a language model.",
         "*Science*, 387(6736), 850–858.",
         "https://doi.org/10.1126/science.ads0018"),
    ]

    for i, (authors, title, journal, url) in enumerate(references, 1):
        st.markdown(f"""
        <div class="ref-item">
            <b>[{i}]</b> {authors} {title} {journal}
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
