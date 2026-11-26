"""Offline inference for the frozen OCDF LiteRT models.

The preprocessing reproduces exactly what the models saw in training:
``tf.image.resize_with_pad(img, 224, 224, method="bilinear")`` on an RGB image with
values 0-255 (no antialiasing, half-pixel centres, zero padding, centred), passed to the
model as float32. Probabilities are ``softmax(logits / T)`` with the frozen temperature
``T``; the prediction is withheld (abstain) when the top probability is below the frozen
threshold. Nothing here opens a network connection.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_SIZE = 224


# --------------------------------------------------------------------------- preprocessing
def _bilinear_weights(out_size: int, in_size: int):
    """Interpolation indices and weights for TF2 bilinear resize (half_pixel_centers=True)."""
    scale = in_size / out_size
    x = (np.arange(out_size) + 0.5) * scale - 0.5
    x_floor = np.floor(x)
    lower = np.clip(x_floor, 0, in_size - 1).astype(np.int64)
    upper = np.clip(np.ceil(x), 0, in_size - 1).astype(np.int64)
    lerp = (x - x_floor).astype(np.float32)
    return lower, upper, lerp


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of an HxWxC array, matching ``tf.image.resize(..., 'bilinear', antialias=False)``."""
    img = img.astype(np.float32)
    in_h, in_w = img.shape[:2]
    y0, y1, wy = _bilinear_weights(out_h, in_h)
    x0, x1, wx = _bilinear_weights(out_w, in_w)
    top = img[y0][:, x0] * (1 - wx)[None, :, None] + img[y0][:, x1] * wx[None, :, None]
    bottom = img[y1][:, x0] * (1 - wx)[None, :, None] + img[y1][:, x1] * wx[None, :, None]
    return top * (1 - wy)[:, None, None] + bottom * wy[:, None, None]


def resize_with_pad(img: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Aspect-preserving resize with centred zero padding, as ``tf.image.resize_with_pad``."""
    h, w = img.shape[:2]
    ratio = max(w / size, h / size)
    rh, rw = int(h / ratio), int(w / ratio)
    resized = resize_bilinear(img, rh, rw)
    out = np.zeros((size, size, img.shape[2]), dtype=np.float32)
    top, left = (size - rh) // 2, (size - rw) // 2
    out[top:top + rh, left:left + rw] = resized
    return out


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as an RGB uint8 array (EXIF orientation applied)."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def preprocess(img: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 [1, 224, 224, 3] in the 0-255 range."""
    return resize_with_pad(img)[None].astype(np.float32)


# --------------------------------------------------------------------------- model registry
@dataclass
class ModelSpec:
    crop: str
    file: Path
    sha256: str
    labels: list[str]
    temperature: float
    threshold: float
    names: dict[str, str] = field(default_factory=dict)


def _labels_json() -> dict:
    return json.loads((REPO_ROOT / "models" / "labels.json").read_text())


def load_spec(crop: str, release: str = "phase1") -> ModelSpec:
    """Read the frozen deployment record for a crop.

    ``release="phase1"`` is the deployed baseline (frozen_choices_v2); ``release="v3"`` is the
    shift-augmented model frozen before the sealed test (frozen_choices_v3).
    """
    if release == "phase1":
        rec = json.loads((REPO_ROOT / "protocol" / "frozen_choices_v2.json").read_text())["deployed_instance"][crop]
        file = REPO_ROOT / "models" / "phase1" / Path(rec["file"]).name
    elif release == "v3":
        rec = json.loads((REPO_ROOT / "protocol" / "frozen_choices_v3.json").read_text())["deployed_instance_v3"][crop]
        file = REPO_ROOT / "models" / "phase2_v3" / Path(rec["file"]).name
    else:
        raise ValueError("release must be 'phase1' or 'v3'")
    names = _labels_json()[crop]
    return ModelSpec(crop, file, rec["sha256"], list(rec["labels_in_output_order"]),
                     float(rec["temperature_T"]), float(rec["threshold"]), names)


def _interpreter(model_path: Path, num_threads: int = 1):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # fall back to the interpreter bundled with TensorFlow
        try:
            from tensorflow.lite import Interpreter  # type: ignore
        except ImportError as exc:
            raise ImportError("Install the LiteRT runtime: pip install ai-edge-litert") from exc
    it = Interpreter(model_path=str(model_path), num_threads=num_threads)
    it.allocate_tensors()
    return it


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Classifier:
    """A frozen OCDF model: load once, then call :meth:`predict` on images."""

    def __init__(self, crop: str, release: str = "phase1", model_path: str | Path | None = None,
                 num_threads: int = 1, verify: bool = True):
        self.spec = load_spec(crop, release)
        path = Path(model_path) if model_path else self.spec.file
        if verify and model_path is None and sha256_file(path) != self.spec.sha256:
            raise ValueError(f"{path.name}: SHA-256 does not match the frozen record")
        self.interpreter = _interpreter(path, num_threads)
        self._in = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]

    def logits(self, x: np.ndarray) -> np.ndarray:
        inp = x
        if self._in["dtype"] in (np.int8, np.uint8):  # full-integer models
            scale, zero = self._in["quantization"]
            inp = np.clip(np.round(x / scale + zero), np.iinfo(self._in["dtype"]).min,
                          np.iinfo(self._in["dtype"]).max).astype(self._in["dtype"])
        self.interpreter.set_tensor(self._in["index"], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self._out["index"]).astype(np.float32)
        if self._out["dtype"] in (np.int8, np.uint8):
            scale, zero = self._out["quantization"]
            out = (out - zero) * scale
        return out[0]

    def predict(self, image: str | Path | np.ndarray) -> dict:
        img = load_image(image) if not isinstance(image, np.ndarray) else image
        z = self.logits(preprocess(img)).astype(np.float64) / self.spec.temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        k = int(p.argmax())
        code = self.spec.labels[k]
        return {
            "crop": self.spec.crop,
            "label": code,
            "name": self.spec.names.get(code, code),
            "confidence": float(p[k]),
            "abstain": bool(p[k] < self.spec.threshold),
            "threshold": self.spec.threshold,
            "probabilities": {self.spec.names.get(c, c): float(v) for c, v in zip(self.spec.labels, p)},
        }
