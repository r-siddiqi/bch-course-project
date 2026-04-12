"""
download and prepare the deeploc 2.0 dataset.

the deeploc 2.0 dataset provides eukaryotic protein sequences with
subcellular localization labels and homology-partitioned train/test splits.

usage:
    python scripts/download_data.py

source:
    https://services.healthtech.dtu.dk/services/DeepLoc-2.0/
"""

import sys
import logging
import urllib.request
import zipfile
import shutil
from pathlib import Path

# add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── dataset urls ───────────────────────────────────────────────────
# deeploc 2.0 dataset (homology-partitioned)
# note: if this url changes, update accordingly
DEEPLOC_URL = (
    "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/"
    "Swissprot_Train_Validation_dataset.csv"
)
DEEPLOC_TEST_URL = (
    "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/"
    "Swissprot_Test_dataset.csv"
)

# alternative: use the hpa-subcellular dataset from huggingface
HF_DEEPLOC_URLS = {
    "train": "https://huggingface.co/datasets/mhcdoctor/deeploc2/resolve/main/train.csv",
    "test": "https://huggingface.co/datasets/mhcdoctor/deeploc2/resolve/main/test.csv",
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """download a file with progress reporting."""
    desc = desc or dest.name
    logger.info(f"downloading {desc} from {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1e6
        logger.info(f"  saved {dest.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        logger.warning(f"  download failed: {e}")
        return False


def download_deeploc():
    """
    attempt to download deeploc 2.0 train/test data.
    tries the official dtu source first, then huggingface mirror.
    """
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    train_path = cfg.data_dir / "deeploc_train.csv"
    test_path = cfg.data_dir / "deeploc_test.csv"

    if train_path.exists() and test_path.exists():
        logger.info("deeploc data already exists, skipping download")
        return

    # try official source first
    logger.info("attempting download from official dtu servers...")
    ok_train = download_file(DEEPLOC_URL, train_path, "deeploc train set")
    ok_test = download_file(DEEPLOC_TEST_URL, test_path, "deeploc test set")

    if ok_train and ok_test:
        logger.info("download from official source complete")
        return

    # fallback: try huggingface mirror
    logger.info("official source unavailable, trying huggingface mirror...")
    if not train_path.exists():
        download_file(HF_DEEPLOC_URLS["train"], train_path, "deeploc train (hf)")
    if not test_path.exists():
        download_file(HF_DEEPLOC_URLS["test"], test_path, "deeploc test (hf)")

    # verify files exist
    if not train_path.exists() or not test_path.exists():
        logger.error(
            "could not download deeploc data automatically.\n"
            "please manually download from:\n"
            "  https://services.healthtech.dtu.dk/services/DeepLoc-2.0/\n"
            "and place train/test csv files in: {cfg.data_dir}"
        )
        sys.exit(1)


def summarize_data():
    """print dataset summary statistics."""
    from collections import Counter

    for split in ["train", "test"]:
        path = cfg.data_dir / f"deeploc_{split}.csv"
        if not path.exists():
            continue

        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # try to find the location column
        cols = [c.lower() for c in rows[0].keys()] if rows else []
        loc_col = None
        for c in rows[0].keys():
            if c.lower() in ("location", "localization", "label", "membrane"):
                loc_col = c
                break

        logger.info(f"\n{'='*50}")
        logger.info(f"{split.upper()} SET: {len(rows)} sequences")
        logger.info(f"columns: {list(rows[0].keys()) if rows else 'empty'}")

        if loc_col and rows:
            counts = Counter(row[loc_col] for row in rows)
            logger.info(f"class distribution ({loc_col}):")
            for cls, n in counts.most_common():
                logger.info(f"  {cls:30s}: {n:5d} ({100*n/len(rows):.1f}%)")


if __name__ == "__main__":
    download_deeploc()
    summarize_data()
    logger.info("\ndata preparation complete!")
