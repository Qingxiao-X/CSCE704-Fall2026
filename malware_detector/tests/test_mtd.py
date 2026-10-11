"""Unit tests for Moving Target Defense model selection.

These tests intentionally avoid loading/training real malware models.
ModelWrapper.load is monkeypatched so the tests are fast, deterministic,
and suitable for local development and CI.
"""

import numpy as np
import pytest

import src.mtd as mtd_module
from src.mtd import MTDSelector


class FakeModel:
    """Minimal stand-in for ModelWrapper used by MTDSelector tests."""

    def __init__(self, probability: float, threshold: float = 0.5):
        self.probability = probability
        self.threshold = threshold

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.full(X.shape[0], self.probability, dtype=float)


def _fake_models():
    return {
        "models/lightgbm": FakeModel(0.90),
        "models/xgboost": FakeModel(0.25),
        "models/random_forest": FakeModel(0.65),
    }


def _patch_model_loader(monkeypatch, models):
    """Replace ModelWrapper.load with a small in-memory fake loader."""

    def fake_load(model_dir):
        if model_dir not in models:
            raise FileNotFoundError(f"Unknown fake model directory: {model_dir}")
        return models[model_dir]

    monkeypatch.setattr(mtd_module.ModelWrapper, "load", staticmethod(fake_load))


def test_mtd_requires_at_least_one_model():
    """An empty MTD model pool should fail clearly."""
    with pytest.raises(ValueError, match="at least one model"):
        MTDSelector([])


def test_mtd_loads_all_requested_models(monkeypatch):
    """Every directory supplied to MTDSelector should be loaded."""
    models = _fake_models()
    _patch_model_loader(monkeypatch, models)

    model_dirs = list(models.keys())
    selector = MTDSelector(model_dirs, seed=42)

    assert len(selector.models) == 3

    loaded_names = [name for name, _ in selector.models]
    assert loaded_names == model_dirs


def test_select_model_returns_only_models_in_pool(monkeypatch):
    """Random selection must never return a model outside the configured pool."""
    models = _fake_models()
    _patch_model_loader(monkeypatch, models)

    selector = MTDSelector(list(models.keys()), seed=42)

    valid_names = set(models.keys())

    for _ in range(100):
        selected_name, selected_model = selector.select_model()

        assert selected_name in valid_names
        assert selected_model is models[selected_name]


def test_selection_is_reproducible_with_same_seed(monkeypatch):
    """Using the same random seed should produce the same model sequence."""
    models = _fake_models()
    _patch_model_loader(monkeypatch, models)

    model_dirs = list(models.keys())

    selector_a = MTDSelector(model_dirs, seed=123)
    selector_b = MTDSelector(model_dirs, seed=123)

    sequence_a = [selector_a.select_model()[0] for _ in range(20)]
    sequence_b = [selector_b.select_model()[0] for _ in range(20)]

    assert sequence_a == sequence_b


def test_random_selection_uses_multiple_models(monkeypatch):
    """Across many selections, random MTD should not remain stuck on one model."""
    models = _fake_models()
    _patch_model_loader(monkeypatch, models)

    selector = MTDSelector(list(models.keys()), seed=42)

    selected_names = {
        selector.select_model()[0]
        for _ in range(100)
    }

    # With the fixed seed and 100 draws from three models,
    # all three should be represented.
    assert selected_names == set(models.keys())


def test_predict_proba_returns_selected_model_and_probability(monkeypatch):
    """MTD prediction should expose which model was used and its probability."""
    models = _fake_models()
    _patch_model_loader(monkeypatch, models)

    selector = MTDSelector(list(models.keys()), seed=42)

    X = np.zeros((1, 10), dtype=float)
    result = selector.predict_proba(X)

    assert set(result.keys()) == {"model_name", "model", "probability"}
    assert result["model_name"] in models
    assert result["model"] is models[result["model_name"]]

