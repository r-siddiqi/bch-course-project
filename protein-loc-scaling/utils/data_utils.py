"""
data loading and preprocessing for the deeploc 2.0 dataset.
handles fasta parsing, label encoding, and sequence filtering.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder

import config as cfg

logger = logging.getLogger(__name__)


def parse_fasta(path: Path) -> list[tuple[str, str, str]]:
    """
    parse a deeploc-style fasta file.
    expects headers like: >ACCESSION LOCATION
    returns list of (accession, location, sequence) tuples.
    """
    records = []
    acc, loc, seq_parts = None, None, []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # save previous record
                if acc is not None:
                    records.append((acc, loc, "".join(seq_parts)))
                # parse header — format: >ACC LOCATION or >ACC\tLOCATION
                parts = line[1:].split()
                acc = parts[0]
                loc = " ".join(parts[1:]) if len(parts) > 1 else "Unknown"
                seq_parts = []
            else:
                seq_parts.append(line)

    # don't forget last record
    if acc is not None:
        records.append((acc, loc, "".join(seq_parts)))

    logger.info(f"parsed {len(records)} records from {path.name}")
    return records


def load_deeploc_csv(path: Path) -> list[tuple[str, str, str]]:
    """
    load deeploc data from csv format.
    expects columns: id, sequence, location (or similar).
    """
    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        # normalize column names to lowercase
        for row in reader:
            row = {k.lower().strip(): v for k, v in row.items()}
            acc = row.get("id", row.get("accession", row.get("protein_id", "")))
            seq = row.get("sequence", row.get("seq", ""))
            loc = row.get("location", row.get("localization", row.get("label", "")))
            if acc and seq and loc:
                records.append((acc, loc, seq))

    logger.info(f"loaded {len(records)} records from csv: {path.name}")
    return records


def load_deeploc(
    data_dir: Optional[Path] = None,
    split: str = "train",
) -> list[tuple[str, str, str]]:
    """
    load deeploc 2.0 dataset from data directory.
    tries fasta first, falls back to csv.

    args:
        data_dir: directory containing dataset files.
        split: 'train' or 'test'.

    returns:
        list of (accession, location, sequence) tuples.
    """
    data_dir = data_dir or cfg.data_dir

    # try common file patterns
    patterns = [
        f"deeploc_{split}.fasta",
        f"{split}.fasta",
        f"deeploc_{split}.csv",
        f"{split}.csv",
    ]

    for pattern in patterns:
        path = data_dir / pattern
        if path.exists():
            if path.suffix == ".fasta":
                return parse_fasta(path)
            else:
                return load_deeploc_csv(path)

    # try any fasta/csv file with the split name
    for f in sorted(data_dir.iterdir()):
        if split in f.stem.lower() and f.suffix in (".fasta", ".csv"):
            if f.suffix == ".fasta":
                return parse_fasta(f)
            else:
                return load_deeploc_csv(f)

    raise FileNotFoundError(
        f"no {split} data found in {data_dir}. "
        f"run scripts/download_data.py first."
    )


def filter_sequences(
    records: list[tuple[str, str, str]],
    min_len: int = cfg.min_seq_len,
    max_len: int = cfg.max_seq_len,
    valid_labels: Optional[list[str]] = None,
) -> list[tuple[str, str, str]]:
    """
    filter sequences by length and valid labels.
    removes sequences with non-standard amino acids (X, B, Z, etc.).
    """
    valid_labels = valid_labels or cfg.labels
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")  # standard 20 amino acids

    filtered = []
    n_short, n_long, n_bad_aa, n_bad_label = 0, 0, 0, 0

    for acc, loc, seq in records:
        seq = seq.upper().strip()

        if len(seq) < min_len:
            n_short += 1
            continue
        if len(seq) > max_len:
            n_long += 1
            continue
        if not all(c in valid_aa for c in seq):
            n_bad_aa += 1
            continue
        if loc not in valid_labels:
            n_bad_label += 1
            continue

        filtered.append((acc, loc, seq))

    logger.info(
        f"filtering: kept {len(filtered)}/{len(records)} | "
        f"short={n_short}, long={n_long}, "
        f"bad_aa={n_bad_aa}, bad_label={n_bad_label}"
    )
    return filtered


def encode_labels(
    locations: list[str],
    le: Optional[LabelEncoder] = None,
) -> tuple[np.ndarray, LabelEncoder]:
    """
    encode string location labels to integer indices.
    if no encoder provided, fits a new one using cfg.labels order.

    returns:
        (encoded_array, label_encoder)
    """
    if le is None:
        le = LabelEncoder()
        le.fit(cfg.labels)

    y = le.transform(locations)
    return y, le


def split_by_partition(
    records: list[tuple[str, str, str]],
    partition_file: Path,
) -> tuple[list, list]:
    """
    split records using a pre-defined partition file
    (e.g., deeploc homology-partitioned splits).
    the partition file should list test set accessions, one per line.
    """
    test_ids = set()
    with open(partition_file) as f:
        for line in f:
            test_ids.add(line.strip().split()[0])

    train = [(a, l, s) for a, l, s in records if a not in test_ids]
    test = [(a, l, s) for a, l, s in records if a in test_ids]

    logger.info(f"partition split: train={len(train)}, test={len(test)}")
    return train, test
