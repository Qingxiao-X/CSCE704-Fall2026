"""Dataset assembly: label parsing, feature extraction over a directory,
duplicate-hash detection, and leakage-safe stratified splitting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features import extract_features_from_path
from src.utils import sha256_of_file

# Label strings accepted in the CSV, mapped to {0, 1}.
_LABEL_MAP = {
    "0": 0, "1": 1,
    "benign": 0, "malware": 1,
    "clean": 0, "malicious": 1,
    "goodware": 0, "bad": 1,
    "safe": 0, "virus": 1,
}


class LabelParseError(Exception):
    pass


def normalize_label(value) -> int:
    s = str(value).strip().lower()
    if s not in _LABEL_MAP:
        raise LabelParseError(
            f"Unrecognized label value: {value!r}. Supported: {sorted(set(_LABEL_MAP))}"
        )
    return _LABEL_MAP[s]


def load_labels_csv(labels_path: str) -> pd.DataFrame:
    """Load a labels CSV. Expects a filename-like column and a label column.

    Accepted filename columns (first match wins): filename, file, path, sample.
    Accepted label columns (first match wins): label, class, malware, target.
    """
    df = pd.read_csv(labels_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    filename_col = next((c for c in ("filename", "file", "path", "sample") if c in df.columns), None)
    label_col = next((c for c in ("label", "class", "malware", "target") if c in df.columns), None)

    if filename_col is None or label_col is None:
        raise LabelParseError(
            f"labels CSV must contain a filename column (one of filename/file/path/sample) "
            f"and a label column (one of label/class/malware/target). Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "filename": df[filename_col].astype(str).str.strip(),
            "label_raw": df[label_col].astype(str).str.strip(),
        }
    )
    out["label"] = out["label_raw"].apply(normalize_label)
    return out[["filename", "label"]]


@dataclass
class DatasetReport:
    n_labels: int = 0
    n_found: int = 0
    n_missing_files: int = 0
    missing_files: List[str] = field(default_factory=list)
    n_unreadable: int = 0
    unreadable_files: List[str] = field(default_factory=list)
    n_duplicate_hashes: int = 0
    duplicate_hash_groups: Dict[str, List[str]] = field(default_factory=dict)
    n_label_conflicts: int = 0
    conflicting_hashes: List[str] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Labels in CSV: {self.n_labels}",
            f"Files found on disk: {self.n_found}",
            f"Missing files: {self.n_missing_files}",
            f"Unreadable/errored files: {self.n_unreadable}",
            f"Duplicate content hashes (same sample appears >1x): {self.n_duplicate_hashes}",
            f"Label conflicts on duplicate hashes: {self.n_label_conflicts}",
            f"Class counts: {self.class_counts}",
        ]
        return "\n".join(lines)


@dataclass
class Sample:
    filename: str
    sha256: str
    label: int
    vector: List[float]


def build_dataset(samples_dir: str, labels_path: str) -> Tuple[List[Sample], DatasetReport]:
    labels_df = load_labels_csv(labels_path)
    report = DatasetReport(n_labels=len(labels_df))

    samples: List[Sample] = []
    hash_to_filenames: Dict[str, List[str]] = {}
    hash_to_labels: Dict[str, set] = {}

    for _, row in labels_df.iterrows():
        filename = row["filename"]
        label = int(row["label"])
        path = os.path.join(samples_dir, filename)

        if not os.path.isfile(path):
            report.n_missing_files += 1
            report.missing_files.append(filename)
            continue

        try:
            result = extract_features_from_path(path)
        except Exception as e:
            report.n_unreadable += 1
            report.unreadable_files.append(f"{filename}: {type(e).__name__}: {e}")
            continue

        if result.status != "ok" or result.vector is None:
            report.n_unreadable += 1
            report.unreadable_files.append(f"{filename}: status={result.status} ({result.error})")
            continue

        report.n_found += 1
        digest = result.sha256
        hash_to_filenames.setdefault(digest, []).append(filename)
        hash_to_labels.setdefault(digest, set()).add(label)

        samples.append(Sample(filename=filename, sha256=digest, label=label, vector=result.vector))

    for digest, names in hash_to_filenames.items():
        if len(names) > 1:
            report.n_duplicate_hashes += 1
            report.duplicate_hash_groups[digest] = names
        if len(hash_to_labels[digest]) > 1:
            report.n_label_conflicts += 1
            report.conflicting_hashes.append(digest)

    labels_arr = [s.label for s in samples]
    report.class_counts = {
        "benign(0)": labels_arr.count(0),
        "malware(1)": labels_arr.count(1),
    }

    return samples, report


def dedup_by_hash(samples: List[Sample], seed: int = 42) -> List[Sample]:
    """Collapse duplicate-hash samples to one representative each, preferring
    the majority label if there is a rare label conflict (and warning is the
    caller's responsibility via DatasetReport)."""
    rng = np.random.default_rng(seed)
    by_hash: Dict[str, List[Sample]] = {}
    for s in samples:
        by_hash.setdefault(s.sha256, []).append(s)

    deduped = []
    for digest, group in by_hash.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        labels = [g.label for g in group]
        majority = 1 if sum(labels) * 2 > len(labels) else 0
        idx = int(rng.integers(0, len(group)))
        rep = group[idx]
        deduped.append(Sample(filename=rep.filename, sha256=rep.sha256, label=majority, vector=rep.vector))
    return deduped


def stratified_split(
    samples: List[Sample], test_size: float = 0.2, seed: int = 42
) -> Tuple[List[Sample], List[Sample]]:
    """Stratified split on de-duplicated (by hash) samples, so no content
    hash can appear in both train and validation."""
    deduped = dedup_by_hash(samples, seed=seed)

    from sklearn.model_selection import train_test_split

    labels = [s.label for s in deduped]
    if len(set(labels)) < 2:
        raise ValueError(
            "Cannot stratify: validation split requires both classes present after "
            "deduplication. Provide more labeled samples covering both classes."
        )

    train, val = train_test_split(
        deduped, test_size=test_size, random_state=seed, stratify=labels
    )
    return train, val
