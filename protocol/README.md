# Protocol and audit trail

These files are the decisions of the study, written and hashed at the moment they were made.
They are released unchanged; do not edit them. `scripts/verify_artifacts.py` checks every
hash and link below.

| Order | File | Written | What it fixes |
|---|---|---|---|
| 1 | `protocol_v2.json` | before training | data protocol: label map, grouping, partitions, class weights |
| 2 | `frozen_choices_v2.json` (+ `.sha256`) | 2026-11-04 20:36 UTC, before any test image was loaded | Phase 1 architecture, deployed instance, format, temperatures, thresholds |
| 3 | `protocol_v3_registration.json`, `protocol_v3_registration_text.md` (+ `.sha256`) | 2026-11-04 22:53 UTC, before any Phase 2 experiment | Phase 2 hypotheses H1–H5, data, decision rules, primary endpoint; cites the hash of (2) |
| 4 | `v3_adaptation_registered.json` | before the sealed test | the adaptation method and image count chosen by the registered rule |
| 5 | `rule1_model_choice.json`, `frozen_choices_v3.json` (+ `.sha256`) | 2026-11-05 14:52 UTC | Phase 2 model, format and abstention rules; cites the hash of (3) |
| 6 | `d5_unsealed.json` | 2026-11-05 16:23 UTC | the single opening of the sealed KaraAgro test half; cites (3) and (5) |
| — | `protocol_v3_deviations.json` | as they occurred | every departure from the registration, its date, reason and effect |

## Deviations in brief

1. A generator bug left the registration's free-text field empty in the JSON; the full text was
   written to `protocol_v3_registration_text.md` with its hash before any Phase 2 training.
2. The registered background-segmentation method assumed plain studio backgrounds; CCMT mixes
   paper and field photographs. It was replaced before any H2 endpoint was read, and H2 is
   reported for paper-background images only.
3. Latency was measured on hosted phones (Qualcomm AI Hub) instead of a phone over USB; the
   endpoint definitions are unchanged.

The full entries are in `protocol_v3_deviations.json`.
