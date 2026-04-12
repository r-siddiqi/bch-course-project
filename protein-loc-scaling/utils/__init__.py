"""utility subpackage — data loading, plotting, evaluation helpers."""

from utils.data_utils import load_deeploc, filter_sequences, encode_labels
from utils.plot_utils import (
    plot_scaling_curve,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_embedding_space,
    plot_per_class_mcc,
)
from utils.eval_utils import (
    compute_metrics,
    bootstrap_ci,
    wilcoxon_compare,
)
