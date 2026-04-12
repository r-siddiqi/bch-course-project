# Protein Subcellular Localization Prediction Using ESM-2 Embeddings

**How Small Can You Go?** — Investigating whether scaling trends observed in regression (Vieira et al., 2025) hold for classification tasks using protein language model embeddings.

## Overview

This project applies ESM-2 embeddings from four model sizes (8M, 35M, 150M, 650M parameters) to predict protein subcellular localization across 10 eukaryotic compartments. We evaluate whether nonlinear classifiers (Random Forest, SVM, XGBoost, MLP) can compensate for lower-quality embeddings from smaller models, or whether larger models consistently dominate.

## Pipeline

```
1. Download data     →  scripts/download_data.py
2. Extract embeddings →  scripts/extract_embeddings.py
3. Train classifiers  →  scripts/train_classifiers.py
4. Generate figures   →  scripts/visualize.py
```

## Quick Start

```bash
# install dependencies
pip install -r requirements.txt

# download deeploc 2.0 dataset
python scripts/download_data.py

# extract embeddings (start with smallest model)
python scripts/extract_embeddings.py --model esm2_8m --split both --device cpu

# train all classifiers on those embeddings
python scripts/train_classifiers.py --model esm2_8m

# generate figures
python scripts/visualize.py
```

For GPU-accelerated extraction (recommended for 150M and 650M):

```bash
python scripts/extract_embeddings.py --model esm2_650m --split both --device cuda
```

## Dataset

**DeepLoc 2.0** — Curated eukaryotic protein sequences with homology-partitioned train/test splits and 10 subcellular localization labels:

Nucleus, Cytoplasm, Extracellular, Mitochondrion, Cell membrane, Endoplasmic reticulum, Plastid, Golgi apparatus, Lysosome/Vacuole, Peroxisome

## Models

| Name | Parameters | Embedding Dim | HuggingFace ID |
|------|-----------|---------------|----------------|
| ESM-2 8M | 8M | 320 | `facebook/esm2_t6_8M_UR50D` |
| ESM-2 35M | 35M | 480 | `facebook/esm2_t12_35M_UR50D` |
| ESM-2 150M | 150M | 640 | `facebook/esm2_t30_150M_UR50D` |
| ESM-2 650M | 650M | 1280 | `facebook/esm2_t33_650M_UR50D` |

## Classifiers

- **Logistic Regression** — linear baseline with balanced class weights
- **Random Forest** — ensemble of decision trees
- **SVM (RBF kernel)** — nonlinear with balanced class weights
- **K-Nearest Neighbors** — cosine distance, distance-weighted
- **XGBoost** — gradient-boosted trees
- **MLP** — shallow neural network (sklearn)

## Evaluation Metrics

- Macro F1 score (with bootstrap 95% CI)
- Weighted F1 score
- Per-class Matthews Correlation Coefficient
- Multi-class confusion matrices
- One-vs-rest ROC curves and AUC
- Wilcoxon signed-rank tests between model sizes

## Visualizations

- Accuracy/F1 vs model size scaling curves (per classifier)
- Normalized confusion matrices
- PCA, t-SNE, and UMAP embedding space projections
- Per-class MCC grouped bar charts

## Project Structure

```
protein-loc-scaling/
├── config.py                  # paths, hyperparameters, model configs
├── requirements.txt
├── scripts/
│   ├── download_data.py       # fetch deeploc 2.0 dataset
│   ├── extract_embeddings.py  # esm-2 mean-pooled embeddings
│   ├── train_classifiers.py   # grid search + evaluation
│   └── visualize.py           # all figures and stats
├── utils/
│   ├── data_utils.py          # fasta/csv parsing, filtering
│   ├── eval_utils.py          # metrics, bootstrap ci, wilcoxon
│   └── plot_utils.py          # matplotlib/seaborn figures
├── data/                      # (gitignored) downloaded datasets
├── embeddings/                # (gitignored) extracted .npy files
└── results/                   # (gitignored) json + figures
```

## References

- Vieira, L.C., Handojo, M.L. & Wilke, C.O. Medium-sized protein language models perform well at transfer learning on realistic datasets. *Nature Scientific Reports* 15, 21400 (2025).
- Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379, 1123–1130 (2023).
- Thumuluri, V. et al. DeepLoc 2.0: multi-label subcellular localization prediction using protein language models. *Nucleic Acids Research* 50, W228–W234 (2022).

## Author

Rida Siddiqi
