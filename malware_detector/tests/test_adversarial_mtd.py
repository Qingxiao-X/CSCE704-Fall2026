"""Tests that MTD can operate over hardened/adversarially trained model bundles.

The test uses fake model bundles so it stays fast and does not require the
real EMBER dataset or trained production models.
"""

import numpy as np

import src.mtd as mtd_module
from src.mtd import MTDSelector
from src.features import FEATURE_DIM


class FakeHardenedModel:
    def __init__(self, probability: float, threshold: float = 0.5):
        self.probability = probability
        self.threshold = threshold

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.full(X.shape[0], self.probability, dtype=float)


def test_mtd_uses_hardened_model_pool(monkeypatch):
    hardened_models = {
        "models/adversarial_lightgbm": FakeHardenedModel(0.90),
        "models/adversarial_xgboost": FakeHardenedModel(0.80),
        "models/adversarial_random_forest": FakeHardenedModel(0.70),
    }

    def fake_load(model_dir):
        return hardened_models[model_dir]

    monkeypatch.setattr(
        mtd_module.ModelWrapper,
        "load",
        staticmethod(fake_load),
    )

    selector = MTDSelector(list(hardened_models.keys()), seed=42)
    X = np.zeros((1, FEATURE_DIM), dtype=float)

    result = selector.predict_proba(X)

    assert result["model_name"] in hardened_models
    assert result["model"] is hardened_models[result["model_name"]]
    assert 0.0 <= result["probability"] <= 1.0
