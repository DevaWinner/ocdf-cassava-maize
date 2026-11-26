# Run records

What each notebook wrote when it ran, released as produced (large CSV files are gzipped). These
are the evidence behind every number in `docs/RESULTS.md`. Image contact sheets are omitted
because they contain dataset images; everything else is here.

| Folder | Notebook | Key files |
|---|---|---|
| `nb01/` | data acquisition and inventory | `inventory_all_v1.csv.gz` (every downloaded image: path, label, size, SHA-256, perceptual hash), `release_register_v1.json` (sources, versions, licences, archive checksums) |
| `nb02/` | duplicates, groups, partitions | `manifest_v2.csv.gz` (every retained image with label, group and partition), `exclusions_v2.csv`, `near_duplicate_pairs.csv`, `reconciliation.csv`, `partition_images.csv` |
| `nb03/` | training (18 runs) | `e1_runs.csv`, `e1_by_architecture.csv`, `runs/*/history.csv` (per-epoch curves) |
| `nb04/` | selection, conversion, calibration | `architecture_selection.json`, `budget_check.json`, `conversion_inspection.json`, `threshold_curves.csv`, `frozen_choices_v2.json` |
| `nb05/` | Phase 1 tests | `results_detail.json`, `bootstrap_seed42.csv` (all intervals and paired differences), `results_seed_mean_sd.csv`, `maize_source_analysis.json` |
| `nb06/` | Phase 2 data and sealing | `manifest_v3.csv.gz`, `phase2_counts.csv`, `protocol_v3_registration.json` |
| `nb07/` | H1 transfer matrix | `e11_transfer_matrix.csv`, `e11_runs.csv`, `runs_v3/*/test_predictions.csv` |
| `nb08/` | H2 background counterfactual | `e12_endpoints.json`, `e12_predictions.csv`, `e12_segmentation_summary.json` |
| `nb09/` | H3 adaptation grid | `e13_runs/*.json` (one file per method × start model × N × draw) |
| `nb10/` | H4 selective INT8; Phase 2 freeze | `layer_errors_*.csv` (quantisation debugger), `e14_files_scored.csv`, `frozen_choices_v3.json` |
| `nb11/` | sealed test and final scoring | `final_results.json`, `h3_*.json`, `predictions/pred_<crop>_<model>_<set>.csv` (per-image predictions of every frozen model on every test and challenge set) |
| `nb12/` | on-device latency | `device_benchmark.csv`, `devices.json`, `profile_*.json` (raw Qualcomm AI Hub profiles), `jobs.json` |

Every folder includes `environment_nbXX.json` (Python, TensorFlow, Keras, NumPy, GPU, time).

Prediction files have one row per image: `sample_id, group_id, label, pred, conf, energy`, and
the abstention decisions under each rule. `sample_id` and `group_id` join to the manifests.
