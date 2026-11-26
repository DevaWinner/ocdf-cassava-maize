#!/usr/bin/env python3
"""Diagnose cassava or maize leaf photos offline with the frozen LiteRT models.

Examples:
  python scripts/predict.py --crop cassava examples/images/*.jpg
  python scripts/predict.py --crop maize leaf.jpg --release v3 --json

The model, temperature and abstention threshold are read from the frozen protocol records,
and the model file is checked against its recorded SHA-256 before use. No network access.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocdf.inference import Classifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="leaf photographs (JPEG or PNG)")
    ap.add_argument("--crop", required=True, choices=["cassava", "maize"])
    ap.add_argument("--release", default="phase1", choices=["phase1", "v3"],
                    help="phase1 = deployed baseline (default); v3 = shift-augmented model")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="print one JSON object per image")
    args = ap.parse_args()

    clf = Classifier(args.crop, args.release, num_threads=args.threads)
    if not args.json:
        print(f"model: {clf.spec.file.name}  (T = {clf.spec.temperature}, abstain below {clf.spec.threshold:.2f})\n")
    for path in args.images:
        t0 = time.perf_counter()
        r = clf.predict(path)
        r["image"] = str(path)
        r["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if args.json:
            print(json.dumps(r))
            continue
        verdict = "NOT SURE - refer to an expert" if r["abstain"] else r["name"]
        print(f"{Path(path).name}")
        print(f"  result:      {verdict}")
        print(f"  top class:   {r['name']} ({r['confidence']:.1%})")
        others = ", ".join(f"{k} {v:.1%}" for k, v in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]))
        print(f"  all classes: {others}")
        print(f"  time:        {r['ms']} ms (incl. image loading)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
