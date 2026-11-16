# RecheckLab: progress for research-guide demonstration

4 October 2026. Prototype v0.2, with recorded replay and experimental live generation.

## Research question

Can signals available in an LLM's initial response predict whether one specified recheck will HELP, HURT, or leave correctness unchanged? Does a small reserved set of target-domain labels improve the decision to spend that extra call?

These scores describe the effect of rechecking, not the probability that an initial answer is correct. NO_CHANGE includes wrong-to-wrong outcomes. Skip recheck keeps an answer that may still be wrong.

## Public data and evaluation

Recorded responses come from [SEVRA Recovery Outcomes](https://huggingface.co/datasets/Sajib-006/sevra-recovery-outcomes), pinned revision `10411137fcaa02b3bd549a2285b43ffe315417f6`, by Sajib Acharjee Dip, Dawei Zhou and Liqing Zhang. This project trained an outcome predictor on their released Qwen3-4B logs; it did not generate these responses or reproduce the paper's evaluation. Attribution, underlying benchmark terms and the unchanged dataset card are in `DATA_SOURCES.md` and `data/raw/DATASET_CARD.md`.

The pilot uses 1,999 source maths questions and 1,221 target CommonsenseQA questions. All attempts of the same canonical question stay together across splits, and each question has total weight one. The target test has 733 questions and 1,466 recorded recheck rows. The UI calls existing rollouts 0/1 **Attempt 1/2**; both share the recorded initial response and remain in research data.

| Target-test policy | Accuracy % | Harmful flips % of all questions | Recheck % | Mean replay tokens |
|---|---:|---:|---:|---:|
| Keep initial | 76.13 | 0.00 | 0.00 | 2243.60 |
| Always recheck | 72.85 | 4.71 | 100.00 | 4719.98 |
| Completion rule | 76.67 | 0.00 | 3.41 | 2334.85 |
| Frozen learned gate | 76.67 | 0.00 | 3.41 | 2334.85 |
| Adapted gate: predeclared 200 labels, subset 1 | 76.67 | 0.00 | 3.41 | 2334.85 |

Independent recomputation confirms that the completion rule, frozen gate and displayed adapted gate select exactly the same 25 questions (50 rows). Their thresholds differ, but no target score lies between them. **The learned gate shows no policy advantage over the completion rule on this test.**

The table describes one predeclared adaptation run. The plot instead averages five repeated, overlapping subsets per budget; its bars are subset SD, not confidence intervals. At 200 labels, mean accuracy is **76.59%**, SD **0.18 percentage points**. No tested budget improves mean accuracy over the frozen gate.

Source-development selection chose class-balanced boosting. Target outcome macro-F1 is **0.394** versus **0.323** for the majority baseline; target HURT recall is **zero**. Probabilities are uncalibrated estimated scores. These findings do not establish reliable harm detection or an adaptation benefit.

## Work completed

- Exactly two tabs: **Demo** and **Results**. Recorded examples work offline with read-only initial answers, automatic recorded metadata and delayed revision/outcome reveal. Results keeps five baseline rows, one plot and one collapsed diagnostics section.
- Live questions use real Ollama calls, strict model identity and automatic prompt/generated token counts, completion status, checkpoint digest, executed finalizer state and extraction. The frozen gate uses only initial-call signals and performs zero or one recheck. No manual metadata or answer-key input appears in the UI.
- Session caching and a request lock prevent repeated clicks and overlapping reruns from duplicating calls. Changing question, mode or record hides stale results. Interrupted/failed attempts stay cached; rerunning does not silently retry an uncertain call. Separate browser sessions or server restarts are new attempts.
- Errors retain a validated initial answer when available; a failed selected recheck leaves the initial answer selected and marks the run incomplete. Missing metadata blocks prediction. Partial returned responses remain unvalidated failure evidence.
- Trusted keys for the three supplied checks are in a separate file. They are used only after execution, never passed to the solver or predictor. Arbitrary questions have unknown actual correctness/outcome. Skipped rechecks have unobserved outcomes even when initial correctness can be scored. Trusted-key grading compares extracted final answers; missing or invalid extractions count incorrect rather than receiving semantic grading.
- Original 24 v0.1/raw data/model/report files remain byte-for-byte unchanged. Matching v0.2 model, thresholds, prepared data, evaluation CSVs and adaptation chart were preserved. Live logs are separate from research results.

## Live protocol and functionality checks

Ollama **0.35.1** is connected and **qwen3:4b** is installed with actual digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`. The initially installed Gemma was not used. The first three functionality attempts failed with HTTP 500: Ollama's oversized default context tried to allocate approximately 38 GB of KV memory. No answer, token counts, prediction or correctness was fabricated. Those attempts remain logged. After explicitly supplying and recording a 4096-token context window, three new attempts completed generation and prediction. The log preserves all six attempts under their canonical question groups, with distinct attempt/run IDs and actual protocol settings.

Even with that model tag, public live use is **experimental**: the released checkpoint digest and full prompts are unavailable. The released card specifies a 4096-token initial generation limit and a possible 512-token finalizer. Live collection uses the existing collector's fixed prompts, a 4096-token context window, a 768 generated-token limit per call, temperature 0, requested `think=false`, seeds 42/43, and no finalizer. Context size and generation limit are different settings. The actual responses still contain extensive reasoning, including a `</think>` marker; the requested flag does not establish that reasoning was suppressed. Actual live identity/settings are logged separately; they are never relabeled as the public replay protocol. Custom predictor artifacts must match their saved checkpoint and full collector protocol, including explicit context, or execution is blocked. Legacy context-unspecified collector metadata keeps its original fingerprint and remains readable for offline research.

The supplied checks are `17 × 6`, notebook/pen change, and bat/ball cost, with separate trusted keys 102, ₹35 and ₹5. Their question-only and key files are `data/live_demo_questions.jsonl` and `data/live_demo_keys.jsonl`. These are functionality checks, not an accuracy, calibration or research sample. Actual returned responses, metadata, failures and policy decisions are appended to `outputs/live/functionality_checks.jsonl`.

| Supplied check | Extracted initial/selected answer | Actual prompt + generated tokens | Completion | Gate decision | Separate key grading |
|---|---|---:|---|---|---|
| 17 × 6 | 102 | 39 + 726 = 765 | stop | Skip recheck | Matches 102 |
| Notebook/pen change | 35 | 65 + 482 = 547 | stop | Skip recheck | Matches ₹35 |
| Bat/ball cost | No parsed final answer | 61 + 768 = 829 | length | Skip recheck | Extraction failure |

The third response exhausted its generation budget without a final-answer marker. Its extracted-answer grade is false under the stated format rule; this is not semantic grading of the reasoning text. The UI warns about truncation and retains the actual partial response. The gate skipped even this truncated response: its offline equivalence to the completion rule does not transfer to these live examples. **No live recheck occurred, so all three actual recheck outcomes remain unobserved.** Recheck execution is verified with recorded examples and controlled backend/UI tests, rather than forced on these questions.

Estimated scores in HELP/HURT/NO_CHANGE order are 1.46/8.79/89.75%, 3.09/8.21/88.69%, and 23.16/15.85/60.99%, respectively. Initial call wall times were 95.42, 23.08 and 35.06 seconds; the first included cold model startup. These three timings are not a performance benchmark. Finalizer-used is false in every successful call because this protocol executes no finalizer.

## Verification

**109 tests passed** under Python 3.12.6 (unittest process exit 0). This includes the original 100 checks plus six artifact save/load/reload regressions and three predictor/backend error-separation checks. It covers recorded selection and UI/evaluation metadata parity, Recheck and Skip paths, missing backend/metadata, protocol mismatches, stale outputs, repeated and concurrent clicks, interrupted calls using actual Streamlit state, partial/truncated responses, actual context API options and trusted-key isolation. Mocked fixtures are not newly generated model responses. The latest full log is `outputs/live_checks/predictor_loading_tests.txt`; the earlier 100-test log is `outputs/live_checks/test_output.txt`.

Independent policy verification reproduced all five comparison rows, all 25 adaptation runs and all repeated-subset summaries. `scripts/verify_baseline.py` verified all 24 original files unchanged. The temporary headless Streamlit server returned HTTP 200 for health and app-shell endpoints and was stopped afterward. A separate AppTest with the actual backend verified connected status, enabled question submission and no automatic generation on input changes. Actual backend status/digest/settings are saved in `outputs/live_checks/backend_status.json`. No visual-browser verification is claimed.

## Predictor loading fix

The message “The saved gate must contain an OutcomePredictor” is raised by `validate_artifact` in `rechecklab/predictor.py` when the dictionary's `predictor` value fails the strict class check. The loader already extracted that value correctly. The active artifact is `outputs/v0.2/models/source_gate.joblib`: a schema-v2 dictionary containing `artifact_schema_version`, `predictor`, `threshold`, `harm_weight`, `feature_names`, `feature_schema_version`, `feature_schema_fingerprint`, `class_order`, `protocol`, `adapted_threshold` and `demo_adaptation`. It contains the fitted class `rechecklab.predictor.OutcomePredictor`, wrapping the selected histogram gradient-boosting classifier with 19 features, 100 fitted iterations and source threshold `0.25968145276947807`. The preserved original `models/source_gate.joblib` contains the four original keys and 18 features; it is not the active artifact.

The exact error was reproduced after Streamlit-style eviction of local modules: the resource cache returned an instance of the previous Python class, while validation expected the newly imported class of the same name. Fresh disk deserialization succeeded. The fix caches only replay data and freshly deserializes the predictor against the current canonical class. Production training now uses the same strict writer contract as the live loader. Type, fitted-estimator, schema, feature/class order, threshold, protocol and matching-hash checks are retained; no class-name coercion, feature invention or heuristic fallback is used.

Six tests verify production writer/live loader round trips, invalid types/old schemas/unfitted models/order mismatches, hashes/protocol/corrupt files, retained loader references, and the actual app cache before and after module eviction. Probabilities and thresholds remain identical across reloads. Predictor errors are separated from endpoint-configuration and LLM connection errors in UI, engine and CLI. Changed files: `app.py`, `rechecklab/predictor.py`, `rechecklab/live.py`, `rechecklab/experiment.py`, `scripts/check_live_backend.py`, `tests/test_artifact_loading.py`, `tests/test_live_backend.py`, `tests/test_live_ui.py`, `tests/test_live_cli.py`, `README.md` and this note.

After the fix, a new real Ollama call for “What is 17 × 6?” returned **102**, with **39 prompt + 726 generated = 765 actual tokens**, completion `stop`, and finalizer-used false. The gate chose Skip recheck; the separate supplied key matched afterward, and its unexecuted recheck outcome remains unknown. The complete response and metadata are in `outputs/live/predictor_loader_fix_check.jsonl`; exact paths, loaded types, keys, imports, hashes and backend status are in `outputs/live_checks/predictor_loading_audit.json`. The current classifier hash still matches its original experiment report. Neither the saved classifier nor Qwen was retrained or fine-tuned.

## Limitations and next experiment

The public pilot has one solver, transfer direction and split; rare harmful source outcomes; inherited correctness labels; public extraction errors; incomplete generation metadata; and replay token estimates excluding gate overhead. Same-name local Qwen3 does not prove the public checkpoint matches. Domain and generation shifts make live scores unvalidated. These three functionality checks cannot validate the predictor; skip decisions reveal no counterfactual recheck outcome.

Next, collect independent paired responses with the local Qwen3 checkpoint and a fully recorded fixed protocol, audit answer extraction and trusted labels, then train and evaluate a gate on question-separated source/target splits. Include enough HURT cases, held-out probability calibration and independent split/seed repetitions. Compare against the completion rule before claiming learned-policy or adaptation gains. Live generation should then use that matching collector-trained artifact.

## Windows launch

```powershell
cd "<project directory>"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`. Recorded examples require no backend. For live mode, install/start Ollama and run `ollama pull qwen3:4b`. Full fresh-environment setup, test commands and collection instructions are in `README.md`.

