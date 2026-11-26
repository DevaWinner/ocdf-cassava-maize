# Notebooks

The complete pipeline, in the order it was run on Google Colab. Each notebook:

- mounts Google Drive and works in `/content/drive/MyDrive/ocdf/`;
- checks the SHA-256 of every file it depends on before running;
- resumes from the last finished step after an interruption;
- writes an environment record and its outputs to `ocdf/records/` (released here in `records/`).

| | Notebook | Hardware | Main output |
|---|---|---|---|
| 01 | `01_data_acquisition_and_inventory` | CPU | image inventories, release register |
| 02 | `02_duplicates_grouping_partitions` | CPU | grouped, partitioned manifest; protocol v2 |
| 03 | `03_training` | GPU | 18 trained models |
| 04 | `04_conversion_calibration` | GPU | LiteRT files, temperatures, thresholds; Phase 1 freeze |
| 05 | `05_testing_evaluation` | GPU | Phase 1 results |
| 06 | `06_phase2_data` | CPU | Phase 2 data, splits and sealed set; registration |
| 07 | `07_transfer_matrix` | GPU | H1 transfer matrix; shift-augmented models |
| 08 | `08_background_counterfactual` | GPU | H2 |
| 09 | `09_adaptation` | GPU | H3 adaptation curves |
| 10 | `10_quantisation_freeze` | GPU | H4; Phase 2 freeze |
| 11 | `11_final_tests` | GPU | sealed-test opening; final results |
| 12 | `12_device_benchmark_aihub` | CPU | on-device latency (needs a Qualcomm AI Hub token) |
| 13 | `13_download_full_datasets` | CPU | full public datasets in Drive |

Credentials (Kaggle, Qualcomm AI Hub) are read from Colab Secrets; never paste them into a cell.
See `docs/REPRODUCIBILITY.md` for the order of execution and what to expect.
