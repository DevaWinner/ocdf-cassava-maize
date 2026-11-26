# Data

No dataset images are stored in this repository. The models were trained and tested on five
public datasets, which you can download from their publishers (or with
`notebooks/13_download_full_datasets.ipynb`). The manifests in `records/` list every image used,
with its source path, SHA-256, perceptual hash, label, group and partition, so you can confirm you
have exactly the same files.

| Id | Dataset | Origin | Crop | Images used | Role | Source | Licence |
|---|---|---|---|---|---|---|---|
| D1 | Cassava Leaf Disease Classification (Makerere AI Lab, NaCRRI) | Uganda, farmers' smartphone photos | cassava | 16,820 | development | [Kaggle competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) | competition rules |
| D4 | Machine learning imagery dataset for maize crop, v6.0 (Mduma & Laizer, 2023) | Tanzania, smartphone | maize | 11,367 | development | [doi:10.7910/DVN/GDON8Q](https://doi.org/10.7910/DVN/GDON8Q) | CC0 1.0 |
| D6 | PlantVillage, maize classes (Hughes & Salathé, 2015) | laboratory, plain backgrounds | maize | 3,852 | development | [Mendeley Data tywbtsjrjv](https://data.mendeley.com/datasets/tywbtsjrjv/1) | CC0 1.0 (republication) |
| D3 | CCMT crop pest and disease dataset, raw images (Mensah et al., 2023) | Ghana, DSLR, Oct–Dec 2022 | cassava, maize | 6,069 | external test only | [doi:10.17632/bwh3zbpkpv.1](https://doi.org/10.17632/bwh3zbpkpv.1) | CC BY 4.0 |
| D5 | KaraAgro AI Maize Dataset, v4.1 (Akogo et al., 2022) | Ghana, field | maize | 5,971 | Phase 2 only; half sealed | [doi:10.7910/DVN/CXUMDS](https://doi.org/10.7910/DVN/CXUMDS) | CC0 1.0 |

Notes:

- Only the **raw** CCMT images are used; its augmented release contains transformed copies that
  would leak between partitions.
- D5 uses the registered subset of three archives (`maize_streak_01.rar`, `healthy_01.rar`,
  `faw_01.zip`), each checked against the MD5 its repository publishes.
- Further CCMT folders (cassava brown spot and green mite; cashew; tomato) and KaraAgro fall
  armyworm images form the unfamiliar-input challenge sets for H5.
- The label map is in [`label_map.csv`](label_map.csv). Exclusions and their reasons are
  recorded per image in `records/nb02/exclusions_v2.csv` and `records/nb06/exclusions_v3_extra.csv`.

## Counts after cleaning

From `records/nb02/reconciliation.csv`:

| Source | Downloaded (selected labels) | Unreadable | Exact duplicates removed | Label conflicts quarantined | Retained |
|---|---|---|---|---|---|
| D1 | 16,822 | 0 | 0 | 2 | 16,820 |
| D3 | 6,199 | 18 | 112 | 0 | 6,069 |
| D4 | 11,370 | 1 | 0 | 2 | 11,367 |
| D6 | 3,852 | 0 | 0 | 0 | 3,852 |

Partitions by class are in `records/nb02/partition_images.csv` and `partition_groups.csv`;
Phase 2 splits in `records/nb06/phase2_counts.csv`.

## References

- Akogo, D., et al. (2022). *The KaraAgro AI Maize Dataset* (Version 4.1). Harvard Dataverse. https://doi.org/10.7910/DVN/CXUMDS
- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060.
- Kaggle. (2020). *Cassava leaf disease classification*. https://www.kaggle.com/competitions/cassava-leaf-disease-classification
- Mduma, N., & Laizer, H. (2023). Machine learning imagery dataset for maize crop: A case of Tanzania. *Data in Brief*, 48, 109108. https://doi.org/10.1016/j.dib.2023.109108
- Mensah, P. K., et al. (2023). CCMT: Dataset for crop pest and disease detection. *Data in Brief*, 49, 109306. https://doi.org/10.1016/j.dib.2023.109306
