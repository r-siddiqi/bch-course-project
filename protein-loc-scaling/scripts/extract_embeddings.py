"""
extract esm-2 embeddings from protein sequences using huggingface transformers.

extracts mean-pooled embeddings from the last hidden layer of each esm-2
model variant (8m, 35m, 150m, 650m). sequences are processed individually
(batch_size=1) in float32 for reproducibility, following recommendations
from vieira et al. (2025).

usage:
    python scripts/extract_embeddings.py --model esm2_8m
    python scripts/extract_embeddings.py --model all
    python scripts/extract_embeddings.py --model esm2_650m --device cuda

requirements:
    pip install torch transformers
"""

import sys
import argparse
import logging
import gc
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from utils.data_utils import load_deeploc_csv, parse_fasta, filter_sequences

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_sequences(split: str = "train") -> list[tuple[str, str, str]]:
    """load and filter sequences from the dataset."""
    data_dir = cfg.data_dir

    # try csv first, then fasta
    for pattern in [f"deeploc_{split}.csv", f"{split}.csv",
                    f"deeploc_{split}.fasta", f"{split}.fasta"]:
        path = data_dir / pattern
        if path.exists():
            if path.suffix == ".csv":
                records = load_deeploc_csv(path)
            else:
                records = parse_fasta(path)
            break
    else:
        raise FileNotFoundError(f"no {split} data in {data_dir}")

    # filter by sequence length and valid amino acids
    records = filter_sequences(records)
    logger.info(f"loaded {len(records)} {split} sequences after filtering")
    return records


def extract_embeddings(
    sequences: list[str],
    model_name: str,
    device: str = "cpu",
    batch_size: int = 1,
) -> np.ndarray:
    """
    extract mean-pooled embeddings from the last hidden layer.

    processes sequences one at a time (batch_size=1) to ensure
    reproducibility — batch processing with padding can introduce
    numerical inconsistencies (vieira et al., 2025).

    args:
        sequences: list of amino acid strings.
        model_name: key from cfg.models (e.g., 'esm2_8m').
        device: 'cpu' or 'cuda'.
        batch_size: kept at 1 for reproducibility.

    returns:
        (n_sequences, emb_dim) float32 numpy array.
    """
    hf_id, emb_dim = cfg.models[model_name]

    logger.info(f"loading {model_name} ({hf_id})...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    n_seqs = len(sequences)
    embeddings = np.zeros((n_seqs, emb_dim), dtype=np.float32)

    logger.info(f"extracting embeddings for {n_seqs} sequences...")
    t0 = time.time()

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            # truncate to max length if needed
            seq = seq[:cfg.max_seq_len]

            # tokenize single sequence
            inputs = tokenizer(
                seq,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=cfg.max_seq_len + 2,  # +2 for special tokens
            ).to(device)

            # forward pass — extract last hidden state
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (1, seq_len, emb_dim)

            # mean pooling over sequence positions
            # exclude special tokens (first=<cls>, last=<eos>)
            seq_emb = hidden[0, 1:-1, :].mean(dim=0)  # (emb_dim,)
            embeddings[i] = seq_emb.cpu().numpy()

            # progress logging
            if (i + 1) % 100 == 0 or i == n_seqs - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_seqs - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  [{i+1}/{n_seqs}] "
                    f"{rate:.1f} seq/s, eta {eta:.0f}s"
                )

            # periodic memory cleanup for large models
            if (i + 1) % 500 == 0:
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    elapsed = time.time() - t0
    logger.info(f"extraction complete: {n_seqs} seqs in {elapsed:.1f}s")

    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    accessions: list[str],
    labels: list[str],
    model_name: str,
    split: str,
):
    """save embeddings and metadata to disk."""
    out_dir = cfg.emb_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # save embedding matrix
    emb_path = out_dir / f"{split}_embeddings.npy"
    np.save(emb_path, embeddings)

    # save metadata (accessions and labels)
    meta_path = out_dir / f"{split}_metadata.npz"
    np.savez(
        meta_path,
        accessions=np.array(accessions),
        labels=np.array(labels),
    )

    size_mb = emb_path.stat().st_size / 1e6
    logger.info(f"saved: {emb_path} ({size_mb:.1f} MB)")
    logger.info(f"saved: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="extract esm-2 embeddings for deeploc sequences"
    )
    parser.add_argument(
        "--model", type=str, default="esm2_8m",
        choices=list(cfg.models.keys()) + ["all"],
        help="which esm-2 model to use (default: esm2_8m)",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "test", "both"],
        help="which data split to process (default: train)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="compute device (default: cpu)",
    )
    args = parser.parse_args()

    # determine which models to run
    model_names = list(cfg.models.keys()) if args.model == "all" else [args.model]
    splits = ["train", "test"] if args.split == "both" else [args.split]

    # auto-detect device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda not available, falling back to cpu")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("mps not available, falling back to cpu")
        device = "cpu"

    logger.info(f"using device: {device}")

    for split in splits:
        # load sequences
        records = load_sequences(split)
        accessions = [r[0] for r in records]
        locations = [r[1] for r in records]
        sequences = [r[2] for r in records]

        for model_name in model_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"model: {model_name} | split: {split}")
            logger.info(f"{'='*60}")

            # check if already extracted
            out_path = cfg.emb_dir / model_name / f"{split}_embeddings.npy"
            if out_path.exists():
                logger.info(f"embeddings already exist at {out_path}, skipping")
                continue

            # extract embeddings
            emb = extract_embeddings(sequences, model_name, device)

            # save to disk
            save_embeddings(emb, accessions, locations, model_name, split)

            # cleanup
            del emb
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
