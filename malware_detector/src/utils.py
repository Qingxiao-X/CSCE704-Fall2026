"""Shared helpers: hashing and numeric sanitization.

Kept dependency-free (stdlib + numpy only) so every module can import it
without risk of circular imports.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, List

# Any feature value outside this range is clipped before it can reach the
# model. This keeps a single corrupted/huge field (e.g. a bogus size_of_image
# read from a malformed header) from producing inf/overflow downstream.
MAX_ABS_FEATURE_VALUE = 1.0e12


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(value, default: float = 0.0) -> float:
    """Coerce a value to a finite float, substituting `default` for
    None/NaN/inf/unparseable inputs, and clipping extreme magnitudes."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    if f > MAX_ABS_FEATURE_VALUE:
        return MAX_ABS_FEATURE_VALUE
    if f < -MAX_ABS_FEATURE_VALUE:
        return -MAX_ABS_FEATURE_VALUE
    return f


def sanitize_vector(values: Iterable[float]) -> List[float]:
    """Ensure no NaN/inf ever reaches the model. Applied as the final step
    of feature extraction, and again defensively right before inference."""
    return [safe_float(v) for v in values]
