# Reproducing the work

Three levels, each building on the last.

## Level 1 — verify (minutes, no data, no GPU)

```bash
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py     # checksums, hash chain, timeline, deployed-model hashes
python scripts/reproduce_results.py    # recomputes 10 headline results exactly from saved predictions
python -m pytest -q                    # the same checks plus inference tests
```

`verify_artifacts.py` checks that:

1. every file in `CHECKSUMS.sha256` is unchanged;
2. each protocol record matches the SHA-256 written at the moment it was frozen;
3. each record cites the hash of the one before it: Phase 1 freeze → Phase 2 registration →
   Phase 2 freeze → sealed-test opening → final results;
4. the timestamps run in that order (registered 2026-11-04 22:53 UTC; frozen 2026-11-05
   14:52 UTC; sealed test opened 2026-11-05 16:23 UTC);
5. the deployed `.tflite` files are the ones named, by hash, in the freeze records.

`reproduce_results.py` recomputes macro-F1, accuracy and per-class recall, with group-bootstrap
intervals, for every frozen model on every labelled test set, and compares them with the values
written when the sealed test was opened. All ten match to within 1e-9.

## Level 2 — re-evaluate the released models (hours, datasets, CPU)

1. Download the datasets: `notebooks/13_download_full_datasets.ipynb`, or by hand from the
   sources in `data/README.md`.
2. Rebuild the working copies with the inventory rules in `src/ocdf/inventory.py`
   (EXIF-rotated RGB JPEG, longest side ≤ 512 px, quality 95).
3. Select the test images from the manifests (`records/nb02/manifest_v2.csv.gz`, column
   `partition`; `records/nb06/manifest_v3.csv.gz` for the Phase 2 halves) and run
   `ocdf.inference.Classifier` on them.
4. Score with `src/ocdf/evaluation.py`. Image identifiers and SHA-256 hashes in the manifests
   let you confirm you have the same files.

## Level 3 — rerun the pipeline (days, datasets, GPU)

Run the notebooks in order on Google Colab (A100 or L4). Each one mounts Google Drive at
`/content/drive/MyDrive/ocdf`, checks the hashes of the files it needs, writes a run record, and
resumes after an interruption.

| Notebook | Step | Hardware |
|---|---|---|
| 01 | download, checksum and inventory the four Phase 1 sources | CPU |
| 02 | near-duplicates, groups, partitions; freeze protocol v2 | CPU |
| 03 | 18 training runs (3 architectures × 2 crops × 3 seeds) | GPU |
| 04 | select, convert to LiteRT, calibrate, choose thresholds; freeze | GPU |
| 05 | Phase 1 tests (internal and CCMT) | GPU |
| 06 | Phase 2 data, KaraAgro registered subset, splits, sealing | CPU |
| 07 | H1 transfer matrix (24 runs) and shift-augmented twins | GPU |
| 08 | H2 background counterfactual | GPU |
| 09 | H3 adaptation grid | GPU |
| 10 | H4 quantisation debugger and selective INT8; Phase 2 freeze | GPU |
| 11 | open the sealed test once; final scoring | GPU |
| 12 | latency on hosted phones (Qualcomm AI Hub token required) | CPU |
| 13 | download the complete public datasets | CPU |

Kaggle and Qualcomm AI Hub credentials are read from Colab Secrets (`KAGGLE_API_TOKEN`,
`QAI_HUB_API_TOKEN`); never paste them into a cell.

Exact bit-for-bit retraining is not guaranteed: GPU kernels are not fully deterministic. Expect
the same conclusions and numbers within the seed-to-seed spread reported in `records/nb03/` and
`records/nb07/`. The released models and predictions are the reference.

## Environment

The notebooks ran with Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2 and NumPy 2.4.6
(`requirements-research.txt`; every run record includes an `environment_nbXX.json`).
The scripts in this repository need only NumPy, pandas, Pillow, scikit-learn and the LiteRT
runtime (`requirements.txt`).
