"""Static feature extraction for Windows PE files.

Backed by the official EMBER 2018 feature set (`ember.features.PEFeatureExtractor`,
feature_version=2, 2381 dims) via `src/ember_extractor.py`, which patches only
the parts of `ember`'s 2018-era code that no longer run against current
LIEF/scikit-learn (see that module's docstring for exactly what was changed
and why) -- the actual numeric feature *definitions* are the unmodified
upstream ones.

CRITICAL INVARIANT: the exact same code path (this module) is used both to
build the training matrix and to featurize a file at prediction time.
`FEATURE_SCHEMA` is the single source of truth for feature identity/order/
dimension and is persisted alongside every trained model (see src/model.py).
For the bulk EMBER dataset, training features come from Elastic's own
precomputed `raw_features` JSON vectorized with the identical
`process_raw_features` this module uses for live files (see
src/train_ember.py) -- so training-time and prediction-time vectorization
share the same formula even though training reads pre-parsed JSON and
prediction parses a fresh PE file.

No PE sample is ever executed. Parsing is purely static (LIEF, via `ember`)
plus raw byte-level statistics (entropy/histograms/strings), also computed
without execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.ember_extractor import get_extractor
from src.utils import sanitize_vector, sha256_of_bytes


def _build_schema() -> List[str]:
    names: List[str] = []
    for fe in get_extractor().features:
        for i in range(fe.dim):
            names.append(f"{fe.name}_{i:04d}")
    return names


FEATURE_SCHEMA: List[str] = _build_schema()
FEATURE_DIM: int = len(FEATURE_SCHEMA)


class PEFeatureExtractionError(Exception):
    """Raised for inputs that cannot be treated as a sample at all."""


@dataclass
class ExtractionResult:
    sha256: str
    status: str  # "ok" | "empty_file" | "not_a_pe" | "parse_error"
    vector: Optional[List[float]] = None
    feature_dict: Optional[Dict[str, float]] = None
    error: Optional[str] = None


def extract_features(data: bytes) -> ExtractionResult:
    """Extract the fixed-length EMBER-2018-schema feature vector for one
    sample's raw bytes.

    Never raises for malformed/truncated/non-PE input -- returns an
    ExtractionResult whose `status` explains why `vector` may be None
    (empty_file, not_a_pe) or best-effort (ok, with byte/string-level
    features populated but header/section/import fields zeroed for a file
    that is truncated/packed/encrypted such that LIEF could not parse it
    at all -- EMBER's own FeatureType classes already degrade to defaults
    when `lief_binary is None`).
    """
    digest = sha256_of_bytes(data)

    if len(data) == 0:
        return ExtractionResult(sha256=digest, status="empty_file", error="file is empty")

    if data[:2] != b"MZ":
        return ExtractionResult(sha256=digest, status="not_a_pe", error="missing MZ header")

    try:
        extractor = get_extractor()
        vec = extractor.feature_vector(data)
    except Exception as e:
        return ExtractionResult(
            sha256=digest,
            status="parse_error",
            error=f"{type(e).__name__}: {e}",
        )

    vector = sanitize_vector(float(v) for v in vec)
    feature_dict = dict(zip(FEATURE_SCHEMA, vector))
    return ExtractionResult(sha256=digest, status="ok", vector=vector, feature_dict=feature_dict)


def extract_features_from_path(path: str) -> ExtractionResult:
    with open(path, "rb") as f:
        data = f.read()
    return extract_features(data)
