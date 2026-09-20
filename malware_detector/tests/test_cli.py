"""End-to-end CLI workflow test (train -> evaluate -> predict) via subprocess,
exactly as a grader would invoke the commands from the README.

IMPORTANT: No real labeled malware/benign dataset is available in this
environment. To exercise the CLI plumbing (argument parsing, dataset
loading, stratified/dedup splitting, model save/load, threshold tuning,
CSV output schema) this test builds a small SYNTHETIC dataset by taking the
one legitimate benign PE fixture and appending random-length random-byte
overlays, then assigns labels via an arbitrary synthetic rule tied to the
overlay size. This produces a model that "trains" successfully and can be
evaluated/predicted against -- it is a plumbing test only and must never be
read as a real accuracy result or a real trained malware detector.
"""

import csv
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALID_PE = os.path.join(PROJECT_ROOT, "tests", "fixtures", "valid_sample.exe")


def _run(args, cwd=PROJECT_ROOT):
    return subprocess.run(
        [sys.executable, "-m", args[0]] + args[1:],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _build_synthetic_dataset(root):
    import random

    base = open(VALID_PE, "rb").read()
    samples_dir = os.path.join(root, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    rng = random.Random(42)

    rows = [("filename", "label")]
    for i in range(20):
        overlay = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 100)))
        fname = f"class0_{i:03d}.exe"
        with open(os.path.join(samples_dir, fname), "wb") as f:
            f.write(base + overlay)
        rows.append((fname, "benign"))

    for i in range(20):
        overlay = bytes(rng.getrandbits(8) for _ in range(rng.randint(3000, 5000)))
        fname = f"class1_{i:03d}.exe"
        with open(os.path.join(samples_dir, fname), "wb") as f:
            f.write(base + overlay)
        rows.append((fname, "malware"))

    labels_path = os.path.join(root, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return samples_dir, labels_path


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli_workflow")
    samples_dir, labels_path = _build_synthetic_dataset(str(root))
    model_dir = os.path.join(str(root), "model")

    result = _run([
        "src.train",
        "--samples", samples_dir,
        "--labels", labels_path,
        "--output", model_dir,
        "--backend", "sklearn_rf",
        "--test-size", "0.25",
        "--seed", "42",
    ])
    assert result.returncode == 0, f"train failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    return {
        "root": str(root),
        "samples_dir": samples_dir,
        "labels_path": labels_path,
        "model_dir": model_dir,
    }


def test_train_produces_expected_artifacts(trained_model):
    model_dir = trained_model["model_dir"]
    for fname in [
        "model.joblib", "feature_schema.json", "threshold.json",
        "training_config.json", "metrics.json", "preprocessing.json",
    ]:
        assert os.path.exists(os.path.join(model_dir, fname)), f"missing {fname}"


def test_evaluate_cli(trained_model):
    result = _run([
        "src.evaluate",
        "--model", trained_model["model_dir"],
        "--samples", trained_model["samples_dir"],
        "--labels", trained_model["labels_path"],
    ])
    assert result.returncode == 0, f"evaluate failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "roc_auc" in result.stdout


def test_predict_single_file_cli(trained_model):
    result = _run([
        "src.predict",
        "--model", trained_model["model_dir"],
        "--input", VALID_PE,
    ])
    assert result.returncode == 0, f"predict failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "sha256:" in result.stdout
    assert "status: ok" in result.stdout
    assert "prediction:" in result.stdout


def test_predict_directory_cli(trained_model):
    out_csv = os.path.join(trained_model["root"], "predictions.csv")
    result = _run([
        "src.predict",
        "--model", trained_model["model_dir"],
        "--input", trained_model["samples_dir"],
        "--output", out_csv,
    ])
    assert result.returncode == 0, f"predict dir failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert os.path.exists(out_csv)

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"filename", "sha256", "prediction", "malware_probability", "status"}
    assert len(rows) == 40
    assert all(r["status"] == "ok" for r in rows)


def test_predict_missing_model_dir_fails_cleanly(tmp_path):
    result = _run([
        "src.predict",
        "--model", str(tmp_path / "no_such_model"),
        "--input", VALID_PE,
    ])
    assert result.returncode != 0
    assert result.stderr.strip() != ""
