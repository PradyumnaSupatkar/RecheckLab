# Frozen-gate feasibility and results on additional benchmark records

## Decision

**MATH500 and GSM8K can be scored by the frozen v0.2 gate without retraining or threshold adaptation.** Both saved datasets contain the exact 19 initial-feature inputs, Qwen3-4B model-family label, `active_verify` action, and the same action-description SHA-256 as the v0.2 artifact. The public dataset card and pinned release identify the same SEVRA source, action and generation limits. The strict artifact feature and metadata checks pass. These evaluations are written as exploratory external replay diagnostics; they are not fresh, untouched confirmation or live inference.

## Provenance and compatibility checks

| Check | MATH500 | GSM8K | Evidence / limit |
|---|---|---|---|
| Raw saved records | 1,000 `active_verify` rows; 500 question IDs; 2 rollouts/question | 2,638 rows; 1,319 IDs; 2 rollouts/question | `data/raw/math500.parquet`, `gsm8k.parquet`; `release_manifest.json` |
| Model and action | `qwen3:4b`, `active_verify` | `qwen3:4b`, `active_verify` | All rows match frozen artifact fields |
| Action-description hash | Matches frozen artifact: `4287d749...d726dd9` | Same | Independently SHA-256 checked from actual `action_description` text |
| Feature inputs | All six required inputs and valid `stop`/`length` completion | Same | `features.py:20-31,45-86`; frozen predictor accepted schema |
| Frozen protocol identity | Derived from pinned SEVRA release ID `10411137fcaa02b3bd549a2285b43ffe315417f6` and `active_verify` | Same | `DATASET_CARD.md` and `release_manifest.json` |
| Exact checkpoint / complete prompts | Not published in local record metadata | Not published | The frozen contract itself records this identity limitation; cannot verify exact checkpoint or full prompt equivalence |
| Answer-key evidence | Local derived booleans only; gold text omitted in public release | Same | The dataset card says gold answers are omitted; this run did not regrade upstream outcomes |
| Question overlap | 0 canonical question-hash overlap with source train/dev/test, CSQA target test, MATH500 or GSM8K | 0 with those cohorts | This run checked canonical normalized query hashes; release IDs are retained as provenance too |
| Prior exposure | The released split is designated evaluation; related project work has inspected it | Same | Newly scored here but not newly collected; classify as exploratory diagnostics, not independent confirmation |

The source MATH `train` holdout is a separate cohort from the published `math500` split. It must not be called MATH Level 5. The MATH500 release has a 500-question name/split, but this audit does not independently establish the provenance/licensing or correctness of each upstream grade from gold answer strings. GSM8K and MATH500 also use two recorded intervention rollouts per question; each question contributes weight one in metrics and bootstrap.

## Frozen policy results

| Policy | MATH500 accuracy | Recheck | Avg total recorded tokens | Saved vs always | GSM8K accuracy | Recheck | Avg total recorded tokens | Saved vs always |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Never | 59.00% | 0.00% | 4,312.5 | 46.92% | 93.40% | 0.00% | 1,180.4 | 59.74% |
| Always | 75.50% | 100.00% | 8,124.9 | 0.00% | 93.40% | 100.00% | 2,932.0 | 0.00% |
| Completion | 75.70% | 45.40% | 7,062.9 | 13.07% | 94.47% | 2.88% | 1,331.4 | 54.59% |
| Frozen ML | 75.30% | 43.60% | 6,933.6 | 14.66% | 94.35% | 2.50% | 1,313.1 | 55.22% |
| Oracle, hindsight | 77.70% | 18.70% | 5,017.7 | 38.24% | 94.66% | 1.25% | 1,235.9 | 57.85% |

Question-clustered paired results: completion minus never is +16.70 pp on MATH500 (95% bootstrap CI +13.40 to +20.10) and +1.06 pp on GSM8K (+0.57 to +1.59). Frozen ML minus completion is -0.40 pp on MATH500 (CI -0.90 to 0.00, unadjusted sign-flip p=.252) and -0.11 pp on GSM8K (CI -0.30 to 0.00, p=.496). The gate uses fewer recorded tokens than completion, but this comparison has no predeclared accuracy noninferiority margin and was not independently confirmed. It is not evidence of superior utility.

## Manuscript treatment

Present these as **fixed-artifact, external benchmark replay diagnostics** after the primary source/target historical comparison. Disclose that all data arise from one SEVRA release and one named Qwen3-4B family, exact prompts/checkpoint digest are unavailable, and upstream grades were not independently recomputed. Do not pool their metrics with source-test or CSQA results. Do not tune thresholds or choose a winner on these cohorts. The exact input hashes and results are in `INPUT_SHA256.json`, `RUN_MANIFEST.json`, and `MAIN_POLICY_COMPARISON.csv`.
