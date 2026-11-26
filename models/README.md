# Model card: OCDF cassava and maize leaf-disease classifiers

## Files

| File | Release | Crop | Format | Size | Role |
|---|---|---|---|---|---|
| `phase1/cassava_EfficientNetB0_s42_fp16.tflite` | Phase 1 | cassava | FP16 | 7.8 MiB | **deployed** |
| `phase1/maize_EfficientNetB0_s42_fp16.tflite` | Phase 1 | maize | FP16 | 7.8 MiB | **deployed** |
| `phase1/*_EfficientNetB0_s42_int8.tflite` | Phase 1 | both | full INT8 | 4.7 MiB | failed the accuracy budget; for comparison |
| `phase2_v3/*_shift_EfficientNetB0_fp16.tflite` | Phase 2 | both | FP16 | 7.8 MiB | shift-augmented model frozen before the sealed test |
| `phase2_v3/*_shift_EfficientNetB0_int8_k0.tflite` | Phase 2 | both | full INT8 | 4.7 MiB | for comparison |
| `phase2_v3/*_shift_MobileNetV2_int8.tflite` | Phase 2 | both | full INT8 | 2.6 MiB | smallest comparator |

SHA-256 hashes are in `../CHECKSUMS.sha256` and in the freeze records
(`protocol/frozen_choices_v2.json`, `protocol/frozen_choices_v3.json`), which also hold each
model's temperature and abstention threshold. **Use the Phase 1 FP16 files unless you are
reproducing a Phase 2 comparison**: they performed better on every shifted test set.

## Licence

**Cassava models: non-commercial use only.** The cassava models were trained on the Kaggle
Cassava Leaf Disease Classification data, whose competition rules allow use for non-commercial
purposes only (participation, academic research and education). The cassava `.tflite` files
are therefore released for non-commercial research and education; the CC BY 4.0 grant does not
extend to commercial use of them. The maize models were trained on CC0 data (Tanzania maize,
PlantVillage) and carry no such restriction.

## Intended use

Research on offline plant-disease classification: benchmarking, transfer and adaptation
studies, and as a starting point for models fine-tuned on local, expert-labelled images.
Not intended as a diagnostic or advisory service, and not validated for any particular farm
or region.

## Interface

| | |
|---|---|
| Input | float32 `[1, 224, 224, 3]`, RGB, values 0–255; aspect-preserving bilinear resize with centred zero padding (`tf.image.resize_with_pad`) |
| Output | logits, one per class, in the order of `labels_in_output_order` |
| Post-processing | `p = softmax(logits / T)`; abstain if `max(p) < threshold` |
| Cassava classes | C01 bacterial blight · C02 mosaic disease · C03 healthy — T = 1.0873, threshold 0.85 |
| Maize classes | M01 streak virus · M02 northern leaf blight · M03 grey leaf spot · M04 common rust · M05 healthy — T = 0.8557, threshold 0.50 |

Class names are in `labels.json`. `src/ocdf/inference.py` implements all of the above.

## Training

EfficientNet-B0 pretrained on ImageNet, fine-tuned on Ugandan cassava (Kaggle, 16,820 images)
and on Tanzanian and PlantVillage maize (15,219 images), grouped so near-duplicates never cross
a split. Selected over MobileNetV2 and MobileNetV3-Small by a pre-declared rule; seed 42 named in
advance. Details: `docs/METHODS.md`.

## Performance (Phase 1 deployed files)

| | Cassava | Maize |
|---|---|---|
| Internal test macro-F1 | 0.728 [0.697, 0.757] | 0.956 [0.935, 0.975] |
| Ghana CCMT macro-F1 | 0.559 [0.534, 0.587] | 0.809 [0.782, 0.832] |
| Ghana KaraAgro sealed test macro-F1 | — | 0.690 [0.673, 0.707] |
| Expected calibration error, internal → CCMT | 0.02 → 0.21 | 0.003 → 0.08 |
| Latency, 1 thread: Galaxy S24 / Pixel 3 / laptop | 19.8 / 85.8 / 6.2 ms | 19.8 / 85.8 / 6.2 ms |

## Known limitations and risks

- **Accuracy drops on new sources**, by 0.09 to 0.65 macro-F1 across the source pairs tested,
  and the model's confidence does not signal it. On Ghanaian cassava most missed blight leaves
  were called healthy, the costliest error for a grower.
- **Abstention does not detect unfamiliar inputs**: other diseases, pests and other crops are
  often accepted with high confidence (AUROC 0.52 for cassava).
- **Background shortcut**: in one test source, healthy leaves were mostly photographed on paper.
  Photograph leaves where they grow.
- **Source–class confounding in maize**: streak virus comes from one development source and
  blight, leaf spot and rust only from PlantVillage laboratory images.
- **No Nigerian or other West African farm images with expert labels** were available for
  training or testing.

Recommendation: before any use in a new region, evaluate on a small set of expert-labelled
local images and fine-tune on them; in this study, tens of labelled images per class recovered
most of the cross-source loss.
