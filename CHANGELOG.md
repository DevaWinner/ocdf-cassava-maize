# Changelog

## 1.0.0 — 2026-11-26

First public release.

- Frozen LiteRT models for cassava and maize: Phase 1 deployed FP16 and full-INT8 files, and the
  Phase 2 shift-augmented FP16, full-INT8 and MobileNetV2 comparator files.
- Pipeline notebooks 01–13 (data acquisition to on-device benchmark) and the `ocdf` package.
- Pre-registration, freezes, deviations and the sealed-test opening record, with a verifiable
  SHA-256 hash chain.
- Run records for every notebook, including per-image predictions for every frozen model on
  every test set.
- `scripts/predict.py` (offline inference), `scripts/verify_artifacts.py` (audit trail),
  `scripts/reproduce_results.py` (recomputes the headline results exactly), tests and CI.
