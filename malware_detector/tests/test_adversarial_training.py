"""Unit tests for adversarial-training utilities."""

import numpy as np
import pytest

from src.adversarial_training import (
    combine_training_samples,
    samples_to_xy,
    validate_adversarial_samples,
)
from src.dataset import Sample
from src.features import FEATURE_DIM


def _sample(name: str, label: int, value: float) -> Sample:
    return Sample(
        filename=name,
        sha256=(name.encode("utf-8").hex() + "0" * 64)[:64],
        label=label,
        vector=[float(value)] * FEATURE_DIM,
    )


def test_combine_training_samples_adds_adversarial_samples():
    clean = [
        _sample("clean.exe", 0, 0.0),
        _sample("malware.exe", 1, 1.0),
    ]
    adversarial = [
        _sample("adv1.exe", 1, 2.0),
        _sample("adv2.exe", 1, 3.0),
    ]

    combined = combine_training_samples(clean, adversarial)

    assert len(combined) == 4
    assert combined[:2] == clean
    assert combined[2:] == adversarial


def test_adversarial_samples_must_keep_malware_label():
    bad = [_sample("wrong_label.exe", 0, 2.0)]

    with pytest.raises(ValueError, match="must be labeled malware"):
        validate_adversarial_samples(bad)


def test_samples_to_xy_returns_correct_shapes():
    samples = [
        _sample("a.exe", 0, 0.0),
        _sample("b.exe", 1, 1.0),
        _sample("c.exe", 1, 2.0),
    ]

    X, y = samples_to_xy(samples)

    assert X.shape == (3, FEATURE_DIM)
    assert y.shape == (3,)
    np.testing.assert_array_equal(y, np.array([0, 1, 1]))


def test_samples_to_xy_rejects_empty_input():
    with pytest.raises(ValueError, match="cannot be empty"):
        samples_to_xy([])
