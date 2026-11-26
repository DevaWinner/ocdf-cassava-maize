"""Tests for the released artefacts. Run with: python -m pytest -q"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocdf.inference import load_spec, resize_bilinear, resize_with_pad  # noqa: E402

HAS_RUNTIME = any(importlib.util.find_spec(m) for m in ("ai_edge_litert", "tensorflow"))
EXAMPLES = sorted((ROOT / "examples" / "images").glob("*.png")) + sorted((ROOT / "examples" / "images").glob("*.jpg"))


def run(script):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script)], capture_output=True, text=True)


def test_audit_trail_and_checksums():
    r = run("verify_artifacts.py")
    assert r.returncode == 0, r.stdout + r.stderr


def test_results_reproduce_exactly_from_predictions():
    r = run("reproduce_results.py")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "10 of 10 results reproduced exactly" in r.stdout


def test_resize_matches_half_pixel_bilinear_on_constant_and_ramp():
    flat = np.full((37, 91, 3), 123, np.uint8)
    assert np.allclose(resize_bilinear(flat, 50, 120), 123)
    ramp = np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (8, 1, 3))
    out = resize_bilinear(ramp, 8, 32)  # 2x downsample: half-pixel centres average neighbouring pairs
    assert np.allclose(out[0, :, 0], np.arange(32) * 2 + 0.5)


def test_resize_with_pad_shape_and_centred_padding():
    img = np.full((480, 640, 3), 200, np.uint8)
    x = resize_with_pad(img)
    assert x.shape == (224, 224, 3) and x.dtype == np.float32
    assert x[:28].max() == 0 and x[196:].max() == 0 and np.allclose(x[28:196], 200)


@pytest.mark.parametrize("crop", ["cassava", "maize"])
@pytest.mark.parametrize("release", ["phase1", "v3"])
def test_frozen_specs_point_at_released_files(crop, release):
    spec = load_spec(crop, release)
    assert spec.file.exists()
    assert 0 < spec.temperature < 5 and 0 < spec.threshold < 1
    assert set(spec.labels) == set(spec.names)


@pytest.mark.skipif(not HAS_RUNTIME, reason="LiteRT runtime not installed (pip install ai-edge-litert)")
@pytest.mark.parametrize("crop", ["cassava", "maize"])
def test_offline_prediction_end_to_end(crop):
    from ocdf.inference import Classifier

    clf = Classifier(crop)
    img = np.random.default_rng(0).integers(0, 256, (300, 400, 3)).astype(np.uint8)
    r = clf.predict(img)
    assert r["label"] in clf.spec.labels
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
    assert r["abstain"] == (r["confidence"] < clf.spec.threshold)
    for path in EXAMPLES:
        assert clf.predict(path)["label"] in clf.spec.labels
