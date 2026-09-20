"""Tests for model save/load consistency, feature-order identity between
training and prediction, and src/predict.py's per-file/directory behavior.

A tiny model is fit on synthetic random feature vectors purely to exercise
the plumbing (save/load, CLI wiring, CSV schema). This is NOT a real
malware detector and no accuracy claims are made or tested here.
"""

import os

import numpy as np
import pytest

from src.features import FEATURE_DIM, FEATURE_SCHEMA, extract_features_from_path
from src.model import ModelWrapper
from src.predict import Row, score_file, main as predict_main

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
VALID_PE = os.path.join(FIXTURES, "valid_sample.exe")


def _make_synthetic_model(seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, FEATURE_DIM))
    y = (rng.random(60) > 0.5).astype(int)
    model = ModelWrapper.create(backend="sklearn_rf", seed=seed)
    model.fit(X, y)
    model.threshold = 0.5
    return model


def test_save_load_roundtrip_predictions_match(tmp_path):
    from src.model import TrainingConfig

    model = _make_synthetic_model()
    rng = np.random.default_rng(7)
    X_test = rng.normal(size=(10, FEATURE_DIM))
    probs_before = model.predict_proba(X_test)

    out_dir = str(tmp_path / "model_out")
    config = TrainingConfig(
        backend=model.backend, seed=42, test_size=0.2, target_fpr=0.01,
        feature_dim=FEATURE_DIM, n_train=48, n_val=12,
    )
    model.save(out_dir, config, {"note": "synthetic test model"})

    loaded = ModelWrapper.load(out_dir)
    probs_after = loaded.predict_proba(X_test)

    assert np.allclose(probs_before, probs_after)
    assert loaded.threshold == model.threshold


def test_feature_order_identical_train_vs_predict(tmp_path):
    """The schema saved with a model must exactly equal the schema live in
    src/features.py, and predict-time vectors must be built in that order."""
    from src.model import TrainingConfig

    model = _make_synthetic_model()
    out_dir = str(tmp_path / "model_out")
    config = TrainingConfig(
        backend=model.backend, seed=42, test_size=0.2, target_fpr=0.01,
        feature_dim=FEATURE_DIM, n_train=48, n_val=12,
    )
    model.save(out_dir, config, {})
    loaded = ModelWrapper.load(out_dir)

    assert loaded.feature_schema == FEATURE_SCHEMA

    result = extract_features_from_path(VALID_PE)
    assert len(result.vector) == len(loaded.feature_schema)
    # Must not raise a dimension-mismatch error.
    loaded.predict_proba([result.vector])


def test_dimension_mismatch_raises():
    model = _make_synthetic_model()
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((1, FEATURE_DIM - 5)))


def test_score_file_valid_pe(tmp_path):
    from src.model import TrainingConfig

    model = _make_synthetic_model()
    out_dir = str(tmp_path / "model_out")
    config = TrainingConfig(
        backend=model.backend, seed=42, test_size=0.2, target_fpr=0.01,
        feature_dim=FEATURE_DIM, n_train=48, n_val=12,
    )
    model.save(out_dir, config, {})
    loaded = ModelWrapper.load(out_dir)

    row = score_file(loaded, VALID_PE, "valid_sample.exe")
    assert row.status == "ok"
    assert row.prediction in (0, 1)
    assert 0.0 <= row.probability <= 1.0
    assert len(row.sha256) == 64


def test_score_file_empty_and_nonpe(tmp_path):
    model = _make_synthetic_model()

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    row_empty = score_file(model, str(empty), "empty.bin")
    assert row_empty.status.startswith("empty_file")
    assert row_empty.prediction is None

    nonpe = tmp_path / "notpe.txt"
    nonpe.write_bytes(b"hello world, not a PE file")
    row_nonpe = score_file(model, str(nonpe), "notpe.txt")
    assert row_nonpe.status.startswith("not_a_pe")
    assert row_nonpe.prediction is None


def test_predict_main_missing_input(tmp_path, capsys):
    from src.model import TrainingConfig

    model = _make_synthetic_model()
    out_dir = str(tmp_path / "model_out")
    config = TrainingConfig(
        backend=model.backend, seed=42, test_size=0.2, target_fpr=0.01,
        feature_dim=FEATURE_DIM, n_train=48, n_val=12,
    )
    model.save(out_dir, config, {})

    rc = predict_main(["--model", out_dir, "--input", str(tmp_path / "does_not_exist.exe")])
    assert rc == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_predict_main_directory_mixed_valid_invalid(tmp_path):
    from src.model import TrainingConfig
    import shutil
    import csv

    model = _make_synthetic_model()
    out_dir = str(tmp_path / "model_out")
    config = TrainingConfig(
        backend=model.backend, seed=42, test_size=0.2, target_fpr=0.01,
        feature_dim=FEATURE_DIM, n_train=48, n_val=12,
    )
    model.save(out_dir, config, {})

    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    shutil.copy(VALID_PE, scan_dir / "good.exe")
    (scan_dir / "empty.bin").write_bytes(b"")
    (scan_dir / "junk.txt").write_bytes(b"not a pe file at all")

    out_csv = tmp_path / "predictions.csv"
    rc = predict_main(["--model", out_dir, "--input", str(scan_dir), "--output", str(out_csv)])
    assert rc == 0
    assert out_csv.exists()

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    by_name = {r["filename"]: r for r in rows}
    assert by_name["good.exe"]["status"] == "ok"
    assert by_name["good.exe"]["prediction"] in ("0", "1")
    assert by_name["empty.bin"]["status"].startswith("empty_file")
    assert by_name["empty.bin"]["prediction"] == ""
    assert by_name["junk.txt"]["status"].startswith("not_a_pe")
