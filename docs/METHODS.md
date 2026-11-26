# Methods

This page summarises how the models were built and tested. The notebooks in `notebooks/` are
the authoritative description: each one states its inputs, steps and outputs, and its run
record is in `records/`.

## Design principle: decide before you look

Every choice that could be tuned to a test set (architecture, epochs, format, temperature,
abstention threshold, adaptation method) was fixed on development partitions and written to a
hashed freeze file before any test image was loaded. Phase 2 went further: its hypotheses,
decision rules and primary endpoint were registered before any Phase 2 experiment ran, and half
of one source was sealed until every choice was frozen. Departures from the registration are
recorded with their dates and reasons in `protocol/protocol_v3_deviations.json`.

## Data preparation (F2) — notebooks 01, 02, 06

1. **Inventory.** Every image of every source is decoded and recorded with its path, original
   label, research label, size, SHA-256 and a 64-bit perceptual hash (`src/ocdf/inventory.py`).
   Archives are checked against the checksums their publishers give. Working copies are
   EXIF-rotated RGB JPEGs with the longest side at most 512 px.
2. **Label map.** Original labels are mapped to eight research classes
   (`data/label_map.csv`). Labels that cannot be shown to mean the same condition across
   sources (for example Kaggle "green mottle" and CCMT "green mite") are excluded rather than
   merged.
3. **Near-duplicates and groups.** Exact duplicates and perceptual-hash near-duplicates are
   linked with union–find, together with capture-session hints, into groups
   (`src/ocdf/partition.py`). Thresholds were checked on contact sheets before use. Pairs with
   conflicting labels are quarantined. 37,664 groups resulted; the largest held 172
   near-identical photographs of one session.
4. **Partitions.** Development sources are split by group and stratified by class into
   train 70% / selection 10% / temperature calibration 5% / threshold selection 5% / test 10%.
   No group crosses a partition boundary. The external sources (CCMT, KaraAgro) are never used
   for training, tuning or calibration in Phase 1.

## Training and selection (F3) — notebooks 03, 07

- ImageNet-pretrained MobileNetV2, MobileNetV3-Small and EfficientNet-B0 (Keras), input
  224 × 224 with aspect-preserving resize and zero padding.
- Head: global average pooling, dropout 0.2, dense softmax. Two stages: train the head
  (learning rate 1e-3, up to 5 epochs), then fine-tune the upper backbone (from `block6a` in
  EfficientNet-B0; 1e-5, up to 25 epochs), batch size 32, early stopping with patience 5 on
  selection macro-F1 with the best weights restored. Inverse-frequency class weights, capped at 5.
- Training augmentation: horizontal and vertical flips, rotation up to 18°, brightness and
  contrast ±10%. Batch-normalisation layers of the backbone run in inference mode.
- Three seeds (17, 42, 2026) per architecture and crop: 18 runs.
- **Pre-declared selection rule:** highest mean selection macro-F1; among architectures within
  0.01 of the best, the smallest file. EfficientNet-B0 won for both crops with no rival within 0.01. The
  seed-42 instance was named in advance as the one to deploy and report intervals for.

## Conversion and calibration (F4) — notebooks 04, 10

- LiteRT conversion to FP32, FP16 (weights) and full-integer INT8 (400 training images per crop
  as the representative set). Each file is checked for prediction parity
  against the Keras model.
- **Budget** (pre-declared): at most 10 MiB per INT8 crop model and 20 MiB for both, and at most
  2 points of accuracy and 0.02 macro-F1 lost relative to FP32. The smallest format that met the
  budget on selection data was deployed: FP16.
- **Temperature scaling** (Guo et al., 2017): one temperature per crop, fitted by minimising
  negative log-likelihood on the calibration partition (`src/ocdf/calibration.py`).
- **Abstention:** the model withholds its answer when the top calibrated probability is below
  a threshold chosen on the threshold-selection partition: the highest-coverage threshold with
  selective accuracy ≥ 0.95, coverage ≥ 0.50 and at least 50 accepted groups (frozen values:
  0.85 cassava, 0.50 maize; `protocol/frozen_choices_v2.json`).

## Evaluation (F6) — notebooks 05, 08, 09, 11, 12

- **Metrics** (`src/ocdf/evaluation.py`): accuracy, balanced accuracy, macro-F1 over the
  classes present in a test set (predicting an absent class counts as an error; absent classes
  are "not evaluated", never averaged in as zero), per-class precision and recall, negative
  log-likelihood, Brier score, 15-bin expected calibration error, coverage and selective
  accuracy at the threshold, and risk–coverage curves.
- **Intervals:** percentile intervals from a bootstrap that resamples whole photo groups
  (2,000 resamples, seed 2026); paired differences resample the same groups for both models.
- **Test sets:** the internal test partition; CCMT (Phase 1 external test, and in Phase 2 split
  by group into adaptation and test halves); the sealed KaraAgro test half (confirmatory).
- **Latency:** LiteRT on two Android phones through Qualcomm AI Hub and on a laptop with the
  network disabled; one and four CPU threads; median of timed runs after warm-up (`bench/`).

## Phase 2 experiments

| Id | Question | Design |
|---|---|---|
| E11 / H1 | Is the cross-source drop general? | leave-one-source-out training for every source, three seeds, Phase 1 vs shift-aimed augmentation |
| E12 / H2 | Is it caused by the background? | segment paper-background CCMT leaves (saturation prior + GrabCut), composite onto plain, Ugandan-field and texture backgrounds, compare predictions |
| E13 / H3 | Can unlabelled local images fix it on the device? | AdaBN, Tent, gated self-training and a labelled reference at N = 10–400, three draws; registered winner applied once to the sealed set |
| E14 / H4 | Can selective INT8 meet the budget? | LiteRT quantisation debugger; keep the k worst operations in float, k ∈ {0, 5, 10, 20, 40} |
| E15 / H4 | How fast on real phones? | hosted-phone and offline-laptop latency of every released file |
| H5 | Does abstention survive shift? | per-class thresholds and the energy score (Liu et al., 2020) against challenge sets of unfamiliar diseases and crops |

The primary confirmatory endpoint (H3) was the paired group-bootstrap difference in macro-F1 on
the sealed KaraAgro half between the adapted and unadapted model; it counted as confirmed only
if the 95% interval excluded zero and the estimate was at least +0.02.

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. *NeurIPS*.
- Li, Y., Wang, N., Shi, J., Hou, X., & Liu, J. (2018). Adaptive batch normalization for practical domain adaptation. *Pattern Recognition*, 80.
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully test-time adaptation by entropy minimization. *ICLR*.
- Yun, S., et al. (2019). CutMix: Regularization strategy to train strong classifiers with localizable features. *ICCV*.
- Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). Random erasing data augmentation. *AAAI*.
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2.
