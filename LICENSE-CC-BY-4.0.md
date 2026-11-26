# Licence for models, records, figures and documentation

Copyright (c) 2026 Aniekan Winner Anietie

Everything in this repository that is not source code — the model files in `models/`, the
protocol and run records in `protocol/` and `records/`, the figures in `figures/`, and the
documentation — is licensed under the
**Creative Commons Attribution 4.0 International licence (CC BY 4.0)**:
<https://creativecommons.org/licenses/by/4.0/legalcode>.

You may share and adapt this material for any purpose, including commercially, provided you
give appropriate credit (see `CITATION.cff`), link to the licence, and indicate if changes
were made.

Source code (`src/`, `scripts/`, `tests/`, `notebooks/`, `bench/`) is licensed under the MIT
licence in `LICENSE`.

## Restriction on the cassava models

**Cassava models: non-commercial use only.** The cassava models were trained on the Kaggle
Cassava Leaf Disease Classification data, whose competition rules allow use for non-commercial
purposes only (participation, academic research and education). The cassava `.tflite` files
are therefore released for non-commercial research and education; the CC BY 4.0 grant does not
extend to commercial use of them. The maize models were trained on CC0 data (Tanzania maize,
PlantVillage) and carry no such restriction.

## Third-party data

No dataset images are redistributed here except the two CCMT example images in
`examples/images/` and the CCMT leaves shown in `figures/background_counterfactual_ccmt.png`
and `figures/leaf_masks_ccmt.png`, which are resized or edited from the CCMT dataset
(Mensah et al., 2023, CC BY 4.0). The models were trained on third-party datasets that keep their own terms; see
`data/README.md`. The Kaggle cassava competition data in particular is governed by the
competition rules and must be obtained from Kaggle.
