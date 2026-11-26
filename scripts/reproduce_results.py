#!/usr/bin/env python3
"""Recompute the headline test results from the saved per-image predictions.

No images, GPU or model runtime are needed: every frozen model's predictions on every test
set are stored in ``records/nb11/predictions/``. This script recomputes macro-F1 and accuracy
(with 95% group-bootstrap intervals, 2,000 resamples, seed 2026) and per-class recall, and
compares each number with the value recorded in ``records/nb11/final_results.json`` when the
sealed test was opened.

Usage:  python scripts/reproduce_results.py            # prints a table, exit code 1 on mismatch
        python scripts/reproduce_results.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocdf.evaluation import fast_accuracy, fast_macro_f1, group_bootstrap  # noqa: E402

LABELS = {"cassava": ["C01", "C02", "C03"], "maize": ["M01", "M02", "M03", "M04", "M05"]}
INSTANCE_FILE = {"phase1": "phase1", "v3": "v3"}
TOL = 1e-9


def recompute(crop: str, test_set: str, instance: str) -> dict:
    df = pd.read_csv(ROOT / "records/nb11/predictions" / f"pred_{crop}_{instance}_{test_set}.csv")
    labels = LABELS[crop]
    y = df["label"].map({c: i for i, c in enumerate(labels)}).to_numpy()
    pred = df["pred"].to_numpy().astype(int)
    present = sorted(set(y.tolist()))
    n_out = len(labels)
    groups = df["group_id"].to_numpy()
    f1 = group_bootstrap(lambda idx: fast_macro_f1(y[idx], pred[idx], present, n_out), groups)
    acc = group_bootstrap(lambda idx: fast_accuracy(y[idx], pred[idx]), groups)
    recall = {labels[c]: float(np.mean(pred[y == c] == c)) for c in present}
    return {"n": int(len(y)), "macro_f1": f1, "accuracy": acc, "per_class_recall": recall}


def close(a, b) -> bool:
    return abs(float(a) - float(b)) <= TOL


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write the recomputed results to this file")
    args = ap.parse_args()

    recorded = json.loads((ROOT / "records/nb11/final_results.json").read_text())["labelled_sets"]
    rows, out, ok_all = [], {}, True
    for key, rec in recorded.items():
        crop, test_set, instance = key.split("/")
        r = recompute(crop, test_set, instance)
        out[key] = r
        ok = (r["n"] == rec["n"]
              and close(r["macro_f1"]["estimate"], rec["macro_f1"]["estimate"])
              and all(close(a, b) for a, b in zip(r["macro_f1"]["ci95"], rec["macro_f1"]["ci95"]))
              and close(r["accuracy"]["estimate"], rec["accuracy"]["estimate"])
              and all(close(r["per_class_recall"][c], v) for c, v in rec["per_class_recall"].items()))
        ok_all &= ok
        lo, hi = r["macro_f1"]["ci95"]
        rows.append((crop, test_set, instance, rec["status"], r["n"], f"{r['macro_f1']['estimate']:.3f}",
                     f"[{lo:.3f}, {hi:.3f}]", f"{r['accuracy']['estimate']:.3f}", "match" if ok else "MISMATCH"))

    head = ("crop", "test set", "model", "status", "n", "macro-F1", "95% CI", "accuracy", "vs record")
    widths = [max(len(str(x[i])) for x in rows + [head]) for i in range(len(head))]
    for line in [head] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(line, widths)))
    print(f"\n{sum(r[-1] == 'match' for r in rows)} of {len(rows)} results reproduced exactly from the saved predictions.")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
