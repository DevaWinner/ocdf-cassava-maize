# Protocol v3: pre-registered Phase 2, "Models that travel"

**Project:** Offline On-Device System for Cassava and Maize Disease Detection Using CNNs
**Builds on:** protocol v2 (manifest SHA-256 93463d32…0d97; frozen choices SHA-256 438e865d…7115). Phase 1 results are final and are reported unchanged as the baseline.
**Status:** Written before any Phase 2 run. Notebook 06 writes this text to Google Drive with a timestamp and SHA-256 on its first run; every later notebook checks that hash. This version replaces an earlier draft that was never written to Drive.

## 1. The question

Phase 1 showed that a model scoring 0.85 accuracy on its own data scores 0.54 on the same crop in another country, and that source cues are perfectly separable in its features. Phase 2 asks four things, in order: how general is the drop (H1); what causes it (H2); what can be done about it on an offline device without expert labels (H3); and what the deployable model actually is, measured on a real phone (H4). Abstention under shift (H5) is kept from Phase 1 and extended.

## 2. Honesty conditions

- Phase 1's internal test and CCMT (D3) results have been seen. Results on those images in Phase 2 are **exploratory**. The protection against selection bias is this registration.
- **KaraAgro (D5)** has never been opened. Half of it, split by group with seed 2026 (`d5_test`), is **sealed** and used once, at the end, as the **confirmatory** set for maize. The other half (`d5_adapt`) may be used for adaptation and as a training source in H1.
- D3 is split the same way into `adapt` and `test` halves per crop. `d3_test` halves are never trained on or adapted to in any experiment. Cassava has no untouched source, so cassava results are exploratory throughout; the report says so.
- Choices are made on selection or `adapt` data only. Nothing is tuned on a `test` half, the challenge set or D5.
- Negative results are reported with the same prominence as positive ones.

## 3. Hypotheses, experiments and endpoints

### H1. The drop is general, not a one-off (transfer matrix; E11)
Train on one source, test on the others, every direction.
- **Maize, two classes (streak virus vs healthy),** three field sources: Tanzania (D4), Ghana-CCMT (D3m), Ghana-KaraAgro (D5). Three folds, each holding out one source. Training uses the other two (D4 training partition; `adapt` halves of D3m and D5), selection on their selection subsets, testing on the held-out source's `test` half (for D4, the Phase 1 test partition).
- **Cassava, three classes,** two sources: Uganda (D1) → Ghana (Phase 1 direction, already done) and Ghana (`d3c_adapt`) → Uganda (D1 test partition), the reverse direction.
- Two arms per fold: **Phase 1 augmentation** and **shift-aimed augmentation** (random resized crop area 0.5–1.0, colour jitter ±25% and hue ±0.05, random erasing ≤25% with p=0.5, CutMix p=0.5). EfficientNet-B0, Phase 1 recipe, three seeds. 24 runs.
Endpoints: macro-F1 matrix (in-source vs cross-source), the mean cross-source drop, and whether shift augmentation reduces it (paired by seed). The fold that holds out D5 is confirmatory for maize.

### H2. The drop is caused by background and source cues (counterfactual; E12)
Using Ghana-CCMT images, whose plain studio backgrounds make leaf segmentation reliable (colour thresholding plus GrabCut, masks checked on sheets, images with failed masks excluded and counted):
- **Background-only:** the leaf region replaced by its mean colour. If the Phase 1 model still predicts classes far from chance from the background alone, it reads backgrounds.
- **Background swap:** the same segmented leaf pasted onto (a) a plain white background, (b) a plain dark background, (c) a Ugandan field background taken from D1 training images with the central region replaced, (d) a random non-plant texture. Prediction and confidence are compared across backgrounds for the same leaf.
Endpoints: share of leaves whose predicted class changes when only the background changes; recall of CBB and CMD on the Ugandan-field background versus the plain original; background-only accuracy versus the majority-class rate. Exploratory (D3 seen), but causal in design.

### H3. The gap can be closed on the device without expert labels (adaptation; E13)
Starting from the frozen Phase 1 model (EfficientNet-B0, seed 42) and from the H1 shift-augmented model:
- **AdaBN:** batch-normalisation statistics re-estimated on N unlabelled target images, nothing else changed.
- **TENT:** entropy minimisation of the normalisation layers' scale and shift on the same images.
- **Gated self-training:** pseudo-labels from the frozen abstention rule (temperature and threshold from Phase 1), accepted predictions used to fine-tune the final blocks, two rounds.
- **Labelled few-shot reference:** fine-tuning with N labelled local images, as the upper reference for what labels would buy.
N ∈ {0, 10, 25, 50, 100, 200, 400} images per crop drawn group-wise from the `adapt` halves, three draws each. Evaluated on `d3c_test` (cassava) and `d3m_test` (maize) during development.
Decision rule: the label-free method and N with the highest mean macro-F1 on `d3m_test` is **registered as the v3 adaptation** and applied once to maize D5: adapt on `d5_adapt`, score `d5_test`.
**Primary confirmatory endpoint:** paired group-bootstrap difference (2,000 resamples) in macro-F1 (streak virus and healthy) on `d5_test` between the adapted and the unadapted Phase 1 model. "Confirmed" requires the 95% interval to exclude zero and the estimate to be ≥ +0.02. Secondary: the same on `d3c_test` and `d3m_test` (exploratory); a Ghana-vs-Uganda source probe on cassava features before and after adaptation.

### H4. Full-integer INT8 fails at identifiable layers, and selective quantisation fixes it; on a real phone the model meets the budget (E14, E15)
- LiteRT's quantisation debugger on the EfficientNet-B0 models reports per-layer error. Layers are ranked; the k worst are kept in float for k ∈ {0, 5, 10, 20, 40}, giving a size–accuracy curve on selection data. Decision rule: the smallest k whose INT8 file passes the Phase 1 budgets (≤10 MiB per crop, ≤20 MiB both; loss ≤2 pp accuracy and ≤0.02 macro-F1 versus FP32). MobileNetV2 (shift-augmented, from a B-arm of E11 with three classes/five classes) full-INT8 is the comparator.
- **On-device measurement:** the official LiteRT `benchmark_model` binary run over USB on the student's own Android phone, for FP16, full-INT8 and selective-INT8 files, one and four threads, 50 warm-up and 200 timed runs, three sessions, with the phone model, chipset, RAM, Android version and thermal state recorded. Plus the Mac offline benchmark from Section 3.1.7. No application is built.
Endpoints: layer-error ranking; the first k passing the budget; median and 90th-percentile latency and peak memory on the phone.

### H5. Abstention under shift (E16)
- Per-class thresholds (C1) chosen on the threshold-selection partition as registered in the earlier draft; energy score (C2) with the 95th-percentile rule.
- Challenge set: near-unfamiliar cassava (CCMT brown spot, green mite), near-unfamiliar maize (CCMT fall armyworm, grasshopper, leaf beetle, leaf blight, leaf spot; KaraAgro fall armyworm), far-unfamiliar (CCMT cashew and tomato).
Endpoints: false-acceptance rate and AUROC for max-softmax versus energy; AUROC of each score for predicting correctness on `test` halves and `d5_test`.

## 4. Data added (Notebook 06)
- D5 KaraAgro v4.1 (CC0, doi:10.7910/DVN/CXUMDS), registered subset `maize_streak_01.rar`, `healthy_01.rar`, `faw_01.zip`; MD5 checked.
- CCMT unused folders (same release as D3), per-file SHA-256 checked.
- Same inventory, duplicate and grouping rules as protocol v2. New images that copy any Phase 1 image (exact, or pHash distance ≤6) are excluded.
- Partitions: `d3c_adapt`/`d3c_test`, `d3m_adapt`/`d3m_test`, `d5_adapt`/`d5_test` (group-wise, stratified by label, 50/50, seed 2026); `challenge_near_cassava`, `challenge_near_maize`, `challenge_far`. `d5_test` is sealed until `frozen_choices_v3.json` exists.

## 5. Fixed decision rules
1. v3 deployed model per crop: the H1 shift-augmented EfficientNet-B0 (seed 42) if its selection macro-F1 is within 0.02 of Phase 1, else the Phase 1 model.
2. v3 deployed format: the smallest file passing all budgets on selection data among selective-INT8 (smallest k), MobileNetV2 full-INT8 (only if it is the deployed model), and FP16.
3. v3 adaptation: as registered in H3; applied to D5 once.
4. v3 abstention: C1 if "target met" for every class, else the Phase 1 global rule; C2 reported as a score only.
5. Primary confirmatory endpoint: H3 on `d5_test`. Secondary confirmatory: H1 fold holding out D5.
6. No experiment is added after registration. Deviations go in `protocol_v3_deviations.json` with date and reason.

## 6. Notebooks
| ID | Content | Notebook |
|---|---|---|
| E7 | Phase 2 data, splits, sealing | 06 (CPU) |
| E11 | Transfer matrix, two arms | 07 (GPU) |
| E12 | Background counterfactuals | 08 (GPU) |
| E13 | Adaptation curves | 09 (GPU) |
| E14 | Quantisation debugger, selective INT8, abstention rules, freeze v3 | 10 (GPU) |
| E16 + final | Challenge set, test halves, D5 unsealed last | 11 (GPU) |
| E15 | Phone and Mac benchmarks | local script |
