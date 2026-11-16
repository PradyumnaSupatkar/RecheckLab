"""Render reports from measured CSVs; exclusive creation, no original writes."""
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
s=pd.read_csv(P/'SUMMARY.csv'); c=pd.read_csv(P/'CATEGORY_COUNTS.csv'); p=pd.read_csv(P/'SUBGROUP_POLICIES.csv');u=pd.read_csv(P/'UNCERTAINTY.csv')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def save(name,text):
    with (P/name).open('x',encoding='utf8') as f:f.write(text)

counts=table(['Partition','Questions / rows','HELP / HURT / NO_CHANGE (question equivalents)','HELP after length flag','Share of HELP'],[
 [r.partition,f'{r.questions} / {r.rollout_rows}',f'{r.HELP_question_equivalents:g} / {r.HURT_question_equivalents:g} / {r.NO_CHANGE_question_equivalents:g}',f'{r.HELP_length_question_equivalents:g}',f'{100*r.HELP_length_question_equivalents/r.HELP_question_equivalents:.2f}%'] for r in s.itertuples()])
categorytable=table(['Partition','Observed primary bucket','Rows','Question equivalents'],[[r.partition,r.category,r.rows,f'{r.question_equivalents:g}'] for r in c.itertuples()])
audit='''# Measurement audit — frozen replay, not a reasoning-repair estimate

**Run:** 2026-11-16_01. **Verdict:** NO-GO for the broad artifact claim as the main original paper; see [verdict](IEEE_PHASE3_VERDICT.md). All quantities below are independently recomputed from existing rows unless explicitly called archived. No outcomes were regraded.

## Provenance and scope

The requested `IEEE_RESEARCH_AUDIT.md`, Phase 2 run's `FEASIBILITY_VERDICT.md`, `NOVELTY_ASSESSMENT.md`, `OVERRIDE_OPPORTUNITY.md`, `POLICY_COMPARISON.csv`, and efficiency `evaluation_2026-11-03/TECHNICAL_REPORT.md` exist and were inspected. Phase 2's selected six completed/nonempty training HELP cases were verified directly against prepared source-training responses, not accepted solely from its report. Its CSV supports the archived residual tie; no model was retrained.

Active v0.2 source data are a **SEVRA MATH-training holdout**, not verified official Level 5. MATH500 replay is a different cohort. CommonsenseQA adaptation and test remain separate. GSM8K and MATH500 contain two active-verification outcomes per question. Local v0.3.x/Qwen pilot data are incompatible generation protocols and are excluded from these tables.

`FROZEN_AUDIT_PROTOCOL.md` was written before aggregate non-training analysis. Its observable rules were based on source-training evidence; no test outcomes selected a new parser, feature, threshold or model. Previously exposed tests are descriptive audits, not fresh confirmation. The script initially rejected the development canonical duplicate's different initial transcripts; inspection found two IDs, 925 and 959, with the same canonical question and answer/grades but different text. The script now records that fact while retaining question-level dependence. This bookkeeping amendment did not change labels, cohort rules or a policy.

## Original grading and accounting pipeline

| Component | Exact local location | Finding |
|---|---|---|
| Public label construction | `rechecklab/data.py:34` `transition_label`; `:74` `validate_pairs`; `:259` `public_frames` | HELP = false→true; HURT = true→false; NO_CHANGE = same correctness. Public extracted strings and booleans are inherited, not recomputed by the local live parser. `public_frames` retains active_verify and marks unavailable completion as unknown. |
| Public provenance | `data/raw/DATASET_CARD.md`, generation/limitations; `data.py:17` | Pinned data revision `10411137fcaa02b3bd549a2285b43ffe315417f6`. Gold intentionally omitted. Base limit 4096 plus up to 512 non-reasoning finalizer tokens. Exact upstream grader/extractor revision was not reconstructed. Current upstream GitHub browsing was incomplete; no attribution to a particular upstream regex is warranted. |
| Live FINAL extraction | `rechecklab/study_protocol.py:173` `parse_study_answer` | Last FINAL/FINAL ANSWER marker; choice must fully match one A–E letter with permitted punctuation. Numeric output goes through `normalize_number`. It parses visible content, not thinking, in `normalize_call` at :235. This is NOT the historical public grader. |
| Live numerical grading | `study_protocol.py:188` `normalize_number`, `:215` `grade_answer`; `study_validation.py:24` | Decimal normalization, guarded thousands separators and simple fractions, exact normalized equality, separate precision audit. Not a general symbolic-equivalence or proof checker. Do not apply it indiscriminately to MATH expressions or ordered pairs. |
| Legacy/demo extraction | `rechecklab/collector.py:28` `parse_answer`, `:89` `response_text` | Different fallback behavior: may consider whole response when no marker; accepts narrower numeric syntax. Combined response_text can include thinking. It must not be confused with study parsing or public grading. |
| Completion and features | `rechecklab/features.py:45` validation, `:88` `extract_one`, `:117` `extract_features` | Nineteen initial-only features. Stop is backend termination, not proof of a complete valid answer. Length does not preclude a later finalizer. Nonempty extracted text is not mathematical validity. |
| Grouping | `data.py:39` `question_hash`, `:233` `group_partition`, `:248` `assert_disjoint` | Canonical question grouping protects repeated/duplicate questions. Source development has 401 rows but 400 questions. |
| Metric accounting | `rechecklab/metrics.py:9` `question_weights`, `:22` `policy_metrics` | Weight each row 1/n_q; average rollout outcomes per question, then average questions. Recheck token cost retained even for NO_CHANGE. No best-of-two choice. Tokens include prompt plus generation, not measured latency. |
| Frozen gate | `rechecklab/predictor.py:121` `probabilities`, `:200` `load_artifact`, `:303` `decide_recheck`; `outputs/v0.2/models/source_gate.joblib` | Existing artifact unchanged. No gate fit or threshold selection in this phase. Archived gate results are labelled as such. |

All 8,080 active-verification rows pass transition/HELP/HURT flag and token-sum assertions. Initial answer, correctness and stop status agree within canonical groups. The one source-development duplicate has different transcripts; it receives total weight one, not two independent votes. Question clusters, not rollout rows, are bootstrapped. Raw support and weighted support are both retained.

Local trusted keys join by full canonical question (including option order for CSQA): all 2,442 CSQA and 2,638 GSM8K rows match local validation/test snapshots. MATH has no joined published keys. A consistency check of already-extracted answers finds **zero disagreements in 9,668 supported initial/revised answer comparisons**: 4,590 CSQA single-letter comparisons and 5,078 supported numeric GSM8K comparisons. The remaining 492 comparisons are unsupported syntax and are not declared grading errors. This checks exact/numeric answer agreement, not whether an extractor missed an answer in the transcript. `KEY_CONSISTENCY.csv` retains every check. No keys were sent to any model.

## Six reported cases, plus one explicitly missing extraction

All seven are `source_train`, MATH, `base_done_reason=stop`, `base_finalizer_used=False`, published false→true (HELP). **Published gold is unavailable** in this replay. The revised accepted answer is not a gold key. The evidence CSV includes initial and revised excerpts; `SOURCE_TRAIN_REVIEW.json` preserves the full paired text. The mathematical cross-checks below are analyst checks of the questions, not recovered official keys.

| ID | Initial extracted string | Revised accepted string | Asserted original final answer / evidence | Preliminary classification |
|---|---|---|---|---|
| 253 | `3x³-5x²+8x-35` | `3x^3-5x^2+8x-35` | Final line uses superscripts; preceding expansion yields coefficients 3, −5, 8, −35. | Representation/equivalence suspect; not missing reasoning. |
| 1203 | `<answer>(3-1)</answer>` | `(3-1)` | `Final answer: <answer>(3, -1)</answer>`; substituting gives 15 and 51 in the two equations. | Wrapper/extraction suspect; comma loss itself is not assumed safe normalization. |
| 1226 | `Finalanswer:(0.50.6)` | `(0.50.6)` | Repeated final marker then `(0.5, 0.6)`; substitution gives −1.5 and 4.7. | Retained marker/extraction suspect; ordered-pair punctuation matters. |
| 1355 | `36x³+8x²+27x+6` | `36x^3+8x^2+27x+6` | Explicit one-line final polynomial agrees with direct expansion. No reasoning trace available. | Representation/equivalence suspect; cannot assess reasoning quality. |
| 1373 | `}frac(29,4)` | `frac(29,4)` | Final displayed `\\text{Final answer: } \\frac{29}{4}`; discriminant gives c≤29/4. | LaTeX wrapper/extraction suspect. |
| 1494 | `Finalanswer:-3x^2+9x-11` | `-3x^2+9x-11` | Repeated final marker; coefficient sums −3, 9, −11 agree. | Retained marker/extraction suspect. |
| 572 | empty | `frac(1,4)` | Final marker followed by displayed fraction on later lines; log relation gives x=5, then log625(5)=1/4. | Missing multiline extraction suspect; additional to the six nonempty cases. |

These are final assertions, not merely incidental occurrences of a gold-looking expression. Nonetheless, the exact upstream failure cause is unverified, no second independent human rater participated, and no published MATH gold was recovered. Confidence is high in the visible representation discrepancy; causal attribution remains provisional. All six are **6/162 = 3.70%** of training HELP; including 572 gives **7/162 = 4.32%**. Their selection was conditioned on prior HELP/completion analysis and cannot estimate corpus-wide artifact prevalence.

## Auditable taxonomy: adjudication rules v1

Use two layers. Layer 1 is the unchanged published correctness transition. Layer 2 is an evidence-based primary mechanism label, with secondary flags. The CSV's automated observational buckets are deliberately less committal than the adjudicated mechanisms below.

Primary decision order, applied to the two independently assessed transcripts:

1. **Ambiguous/unclassifiable** takes precedence if the trusted key, final assertion, semantic equivalence, chronology or essential transcript is insufficient or conflicting. Multiple unretracted answers, a correct intermediate followed by a wrong final answer, or a correct conclusion supported by an invalid argument must not be called reasoning repair. Record a specific reason, not a default guess.
2. If the published transition is NO_CHANGE, use **published stable-correct** or **published stable-wrong** in the mechanical audit. After trustworthy adjudication, retain those labels only if that assessment agrees; disagreements return to unresolved measurement discrepancy. Stable-correct does not mean sound reasoning, and stable-wrong does not mean unchanged content.
3. For HELP with a verified, unambiguous correct final assertion already in the initial transcript, use **extraction/equivalence recovery** if the assertion satisfies the declared output contract but extraction/grading missed it; use **format-only recovery** if a known required output contract was violated. If the contract or grader is missing, use **representation recovery unresolved**. No occurrence-only gold matching.
4. For remaining HELP, use **completion recovery** only when the initial trace is actually unfinished or has no final assertion, the revision supplies one, and evidence supports continuity rather than an identified corrected error. Length status alone only merits a secondary truncation flag. A length-limited trace with both a repaired error and new finalization remains ambiguous/mixed unless precedence can be justified.
5. Use **supported answer-content correction** when a valid, completed, unambiguously wrong initial answer changes to a verified correct answer. This still does not prove reasoning repair. Add a **reasoning-error repair supported** secondary flag only if annotators identify a specific invalid step and a corrected valid replacement; absence of a trace remains unassessed.
6. For HURT, mirror the checks: **supported harmful answer-content revision** requires verified correct initial assertion and wrong revised assertion; **output/completion loss** covers a previously correct answer becoming unextractable/incomplete without demonstrated wrong content. Uncertain key or mechanism remains **HURT unresolved**. Do not equate an invalid revised format with a proven wrong reasoning path.

Primary labels are exclusive by this decision order. Secondary flags may overlap: initial/revision length, initial/revision finalizer, missing extraction, malformed/multiple marker, contradiction, unknown completion, representation discrepancy, key missing, reasoning not assessable. Format compliance, mathematical correctness, and reasoning validity are separate axes.

### Annotation procedure before a publishable causal rate

Use written examples from training only. Randomize transcript order and hide published transition, policy selection, model identity and partner transcript during independent per-transcript annotation. Assess claimed final answer and reasoning support before revealing the trusted key; then assess answer equivalence. Pair the independently assessed transcripts only to assign mechanism. Full blinding is impossible when a revision references the initial answer; record that limitation.

Have two independent human annotators assess all 177 training HELP/HURT records plus a fixed random sample of 60 NO_CHANGE records (30 published stable-correct, 30 stable-wrong), with sampling seed recorded in advance. Include the seven selected examples but identify them as known suspects. Keep both raw annotations, disagreements and adjudicator rationale. Report raw agreement and nominal Cohen's kappa, per-class agreement and question-bootstrap intervals; rare categories can make kappa unstable. No agreement score is claimed here. Adjudicate without overwriting either annotation. Use a third qualified adjudicator when available; otherwise retain unresolved. Reserve non-training annotation until guidelines are frozen. Audit HURT symmetrically so recovery is not privileged. A representative population estimate needs sampling weights for sampled controls and must not treat known suspects as a random sample.

## Measured observational counts

'''+counts+'''

Question equivalents mean expected counts under a uniform recorded-rollout draw; they are not fractional distinct people/questions. Raw CSQA test support is 21 HELP / 69 HURT / 1,376 NO_CHANGE, not 10.5 independent HELP questions. Raw development support is 47 / 6 / 348. See `SUMMARY.csv` for every raw count.

'''+categorytable+'''

**Causal unclassifiable fraction:** 100% of HELP/HURT cases remain without a fully validated causal mechanism label under the strict adjudication standard. Within that unresolved set, seven training HELP records have reviewed representation evidence; the remaining buckets describe recorded states. Zero validated reasoning-repair annotations means “not established,” not “none exist.” NO_CHANGE classifications are published grade states only.

Truncation-associated harmful outcomes are also frequent: 9/15 training, 5/6 development, 3/3 source test, 66/76 target adaptation, 57/69 target test, 21/22 MATH500, and 24/33 GSM8K HURT rows have revision `length`. These could include truncation, finalizer behavior and content errors; they are not automatically genuine harmful reasoning revisions.

Initial finalizers were used in 124/154 training length-associated HELP, 35/47 development, 49/62 source test, 12/12 target adaptation, 8/8 target test, 156/179 MATH500 and 24/28 GSM8K. Therefore attributing all length-associated HELP to absent final answers would be demonstrably unjustified.

## Limitations and preservation

No independent human labels, pinned upstream mathematical grader replay, complete finalizer-call provenance, or trustworthy per-call historical latency is available in this analysis. Raw response text may not expose separate hidden/visible channels. Alternate actions in raw MATH training were not used to search for a better method; this audit follows active_verify and cannot answer finalization causality. Do not pool local pilots or exposed test data as independent replication.

Original data are unchanged. `PRESERVATION_CHECK.json` verifies 339 original files. All new work is confined to this directory. The initial six-case observation is credible as a grading-sensitivity warning, but not a population estimate or evidence of deployable accuracy improvement.

## Reproduction

From the project root: `.venv\\Scripts\\python.exe -B outputs/ieee_phase3_measurement/run_2026-11-16_01/audit.py --output outputs/ieee_phase3_measurement/run_reproduction_NEW`. The destination must not exist. This copies the frozen protocol and recomputes the case/summary/category/policy files. `verify_and_supplement.py` and `build_reports.py` operate on their own containing run directory and refuse to overwrite output files; copy those scripts into a new reproduced run if needed. No dependency installation is required. Input hashes, seed, versions and protocol hash are in `RUN_MANIFEST.json`.
'''
save('MEASUREMENT_AUDIT.md',audit)

original=table(['Partition','Initial %','Always %','Difference pp [95% paired question-bootstrap CI]'],[[r.partition,f'{r.initial_accuracy_pct:.4f}',f'{r.always_accuracy_pct:.4f}',f'{r.gain_pp:+.4f} [{r.gain_ci_low_pp:.4f}, {r.gain_ci_high_pp:.4f}]']for r in s.itertuples()])
subrows=[]
for part in s.partition:
    for sub in ['stop_nonempty','strict_FINAL_proxy']:
        g=p[p.partition.eq(part)&p.subgroup.eq(sub)].set_index('policy')
        a=g.loc['never'];b=g.loc['always']
        subrows.append([part,sub,int(a.questions),f'{a.accuracy_pct:.4f}',f'{b.accuracy_pct:.4f}',f'{b.accuracy_pct-a.accuracy_pct:+.4f}',f'{a.average_total_tokens:.1f} / {b.average_total_tokens:.1f}'])
sensitivity='''# Metric sensitivity — no retrospective grade replacement

## A: Frozen original evaluation

All original booleans, labels and keys remain unchanged. This is **always-minus-never** gain, not gate gain and not a causal estimate of semantic reasoning repair.

'''+original+'''

Intervals use 2,000 paired question-cluster resamples, seed 20261116. These describe empirical question variation conditional on this solver/protocol and recorded rollouts; they do not capture systematic grader errors, uncertainty about mechanisms or model-to-model generalization. Previously inspected tests are descriptive. No multiple-comparison significance claim is made.

Independently recomputed completion accuracies: source training 92.2435%, development 92.0000%, source test 90.0000%, target adaptation 78.2787%, target test 76.6712%, MATH500 replay 75.7000%, GSM8K replay 94.4655%. Complete costs, recheck rates and HELP/HURT contributions are in `SUBGROUP_POLICIES.csv`.

Archived frozen-gate results from `outputs/ml_efficiency_evaluation/evaluation_2026-11-03/policy_metrics.csv`: development 92.00% / 4,081.3425 tokens, source test 90.00% / 4,412.825 tokens, target test 76.6712% / 2,334.8465 tokens. The matching completion costs independently recomputed here are 4,121.0125 / 4,402.4200 / 2,334.8465. No fresh ML inference, retraining, or alternative-grade gate evaluation was performed. Nothing here demonstrates learned superiority.

## B: Conservative recovery — bounded diagnostic, not an alternative grader

**Not executed as population regrading.** The seven reviewed training records support narrow representation-recovery suspicions, but authoritative MATH keys and a pinned extraction/equivalence contract are unavailable. A general parser inferred from those positives would risk asymmetric false acceptance. Gold-looking intermediate expressions, option letters in explanations and discarded candidates are ineligible. No proposed alternative correctness values are assigned; `alternative_grade_applied=False` for every case.

Why each proposed review is warranted is logged in `CORRECTION_CASES.csv` and the seven-case table: Unicode exponent representation (253,1355), retained answer wrapper (1203), repeated final marker (1226,1494), retained LaTeX text brace (1373), missing multiline extraction (572). A validated future change must record original answer/grade, exact asserted span, trusted key, permitted transformation, revised grade, adjudicators and rationale, for both initial and revised outputs. Distinguish exact numeric equality from expression equivalence; never drop ordered-pair delimiters as a general rule.

Illustrative arithmetic only: if independent adjudication eventually changed the six nonempty initial grades from false to true while all other grades remained fixed, the training always-minus-never gain would fall by 6/1199 = **0.5004 pp**. Including 572 would subtract **0.5838 pp**. This is a conditional accounting example, not a corrected accuracy result, not a full symmetric audit, and not an estimated population artifact rate. It does not justify correcting unseen cases.

The separate key-consistency check uses the existing live numeric normalizer only on already-extracted GSM8K numeric strings, and exact A–E strings for CSQA. It finds no supported-syntax mismatch; it is **not** conservative transcript recovery and does not validate unsupported outputs.

## C: Completion-controlled analysis

True “valid completed answer” requires adjudication. Available proxies are:

- `stop_nonempty`: stopped initial call and nonempty publisher-extracted answer.
- `strict_FINAL_proxy`: additionally a nonempty FINAL/FINAL ANSWER assertion on the last nonempty transcript line, using the frozen regex. This establishes marker syntax only, not correctness, absence of contradiction, or mathematical validity. The marker rule omits some valid multiline/symbolic formats and must not be advertised as a universal evaluator.

Both use initial-state evidence only. Their denominators differ from the overall cohort. Eligibility is not selected by correctness. Within eligible question groups the recorded outcomes remain equally weighted; all repeated question clusters are retained together for dependence accounting. No better rollout is chosen. These tables are population stratifications, not regraded full-benchmark accuracies.

'''+table(['Partition','Subset','Questions','Never %','Always %','Gain pp','Never / always tokens'],subrows)+'''

Within these subsets completion skips everyone, so its accuracy and cost equal never recheck. This is definitional, not an algorithmic improvement. Training stop/nonempty gain is −0.7625 pp versus +12.2602 pp overall; development is −1.3115 versus +10.25; source test is +0.3413 versus +15.50. The difference is strong evidence that whole-cohort gains depend on initial-state mixture. It does not prove that the length stratum gains are merely formatting, or that controlling initial generation would preserve the same answers.

The full-cohort decomposition is exact: gain = sum over initial-state strata of stratum weight × (HELP−HURT)/stratum question mass. Reporting only completed outputs changes the estimand; do not replace the original whole-cohort number with it. Do not call the strict proxy a causal intervention or a rule improving benchmark scores.

## D: Ambiguity-aware interpretation

Every mechanism remains unresolved unless independently supported. The number of genuine semantic HELP events can range from zero to all published HELP events under current information; reviewed suspects do not create a defensible population-wide causal lower bound. If one restricts content contributions to the *published changing events*, their net contribution can lie in the conservative bounds below. These are identification bounds, **not confidence intervals** or corrected total accuracies; unobserved discrepancies among NO_CHANGE records could broaden an actual regrading analysis further.

'''+table(['Partition','Potential content contribution lower pp','Upper pp','Length-associated HELP % [bootstrap CI]'],[[r.partition,f'{r.unidentified_content_net_lower_pp:.4f}',f'{r.unidentified_content_net_upper_pp:.4f}',f'{r.length_HELP_pct:.2f} [{r.length_HELP_ci_low_pct:.2f}, {r.length_HELP_ci_high_pct:.2f}]']for r in u.itertuples()])+'''

The development ratio bootstrap is degenerate at 100% because all observed HELP have length flags. It is not evidence that the population proportion is exactly 100%; resampling cannot invent unseen categories. These intervals are conditional sampling summaries and do not resolve finalizer confounding or label interpretation.

## Interpretation

There is verified sensitivity to completion-based cohort definition and strong bounded evidence of representation-sensitive grades. There is **no verified alternative grading result** and no measured fraction of genuinely repaired reasoning. A symmetric, blinded manual audit with trusted keys is the minimum next validity step. Changing extraction to maximize reported accuracy would answer the wrong question.
'''
save('METRIC_SENSITIVITY.md',sensitivity)
print('Created MEASUREMENT_AUDIT.md and METRIC_SENSITIVITY.md')
