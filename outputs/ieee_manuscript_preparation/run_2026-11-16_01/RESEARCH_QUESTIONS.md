# Research questions and evidence status

## Paper questions

**RQ1 — Does selective rechecking improve final-answer accuracy over never rechecking in recorded paired responses?**

Descriptively yes for MATH source test (+15.25 pp for completion), the pinned MATH500 replay (+16.70 pp), and GSM8K (+1.06 pp); CSQA completion is +0.55 pp, with sparse discordant outcomes and an unadjusted cluster sign-flip p=.121. These are archived response outcomes, not fresh generation. For MATH/GSM8K, completion costs less recorded total token use than always-recheck (13.07% and 54.59% savings respectively). This shows a replay policy trade-off, not necessarily general utility or wall-time savings.

**RQ2 — Does the learned gate add value over completion-based gating?**

No demonstrated accuracy improvement. It ties on source development, source test and CSQA target test; it spends 10.41 more tokens/question than completion on source test. It differs on nine MATH500 and five GSM8K question clusters: observed accuracy is 0.40 and 0.11 pp lower, respectively, while recorded token use is slightly lower. The target decisions are identical. Available evidence favors the simpler completion heuristic for interpretability and does not establish a gate advantage.

**RQ3 — How does this behavior vary across task domains under the same recorded policy family?**

The magnitude and event mix differ: MATH500 has 18.7% question-equivalent HELP and 2.2% HURT outcomes across two rollouts; GSM8K has 1.25% of each; CSQA target has 1.43% HELP and 4.71% HURT row-equivalents. Completion policy therefore selects very different fractions (45.4%, 2.88%, and 3.41%) and yields different cost/accuracy trade-offs. This is descriptive heterogeneity, not causal attribution to task domain; collection, question mix and grading are inherited from SEVRA.

**RQ4 — How well does the frozen classifier identify HELP / NO_CHANGE / HURT?**

Three-class argmax accuracy ranges from 70.20% on MATH500 to 94.39% on GSM8K, but macro-F1 is only 0.394 on CSQA, 0.466 on MATH500, and 0.496 on GSM8K. The HURT F1 is zero on both historical tests, .033 on MATH500 and .056 on GSM8K. Dominant NO_CHANGE support inflates accuracy and weighted-F1, so event-level recall/precision must accompany them. Exact fixed-model metrics and support are in `CLASSIFIER_METRICS_CONFUSION.csv`.

## Evidence labels

- **Measured in this run:** frozen model predictions/policies over existing prepared/raw parquet files; question-cluster bootstrap intervals; paired sign-flip tests; tokens from saved outcome records; hashes of all evaluated input data and model.
- **Historical/previously exposed:** source test and CSQA target test, and the previously viewed source-development set. Their new calculations validate arithmetic and add cluster intervals; they do not become fresh evaluations.
- **Newly computed external diagnostic:** MATH500 and GSM8K policy/classification metrics using the already-frozen artifact, with zero canonical query overlaps. They remain nonconfirmatory because the data release and project have been inspected and protocol identity is incomplete.
- **Not established:** live latency/energy savings; generality beyond this recorded solver/action; causal effect of domain; gate calibration under new prompts/models; statistically reliable harmful-revision avoidance.

## Selection and repeated testing

No policy or threshold was selected from any evaluated test. The v0.2 threshold was frozen before this run. Source-development values are included only as descriptive evidence and must be labeled model/threshold-selection data in any paper. Target adaptation is not used for a new threshold. All comparisons average within question groups; raw rollout counts are shown alongside question-equivalent quantities. Paired p-values are unadjusted and are supporting diagnostics, not a battery for selecting favorable cohorts.
