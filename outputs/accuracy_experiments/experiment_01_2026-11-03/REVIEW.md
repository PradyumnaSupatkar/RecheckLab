# Experiment 1 review

The experiment did **not** improve source-development accuracy and does not
support promotion. The selected configuration is the existing balanced HGB
classifier, with a different training-selected threshold.

All 11 requested configurations completed. Five outer stratified group folds
each held out three HURT questions; three inner folds selected each outer
threshold. Configuration ranking used question-equal HELP minus HURT, then
average extra recheck tokens. Macro-F1 and log loss did not choose the winner.

The balanced HGB and square-root-balanced logistic regression with C=0.1 tied
at CV utility 152: HGB selected 157 HELP and 5 HURT, while logistic selected
158 HELP and 6 HURT. HGB won the token tie-break, averaging 1,546.274 extra
tokens versus 1,827.289. Its pooled nested-CV accuracy was 92.2435%.
The selected winner's CV score is subject to configuration-selection optimism.

The final threshold, -0.04560480597941055, was selected using only the winner's
source-training OOF predictions. Its optimized training utility is explicitly
marked optimistic in selection.json. The selection and production-compatible
artifact were saved before source development was loaded. No decision was
changed after viewing development results.

| Source-development policy | Accuracy | Helpful fixes | Harmful flips | Recheck | Average total tokens |
|---|---:|---:|---:|---:|---:|
| Selected experiment | 91.75% | 47 | 3 | 78.50% | 5,070.836 |
| Existing frozen gate | 92.00% | 47 | 2 | 23.25% | 4,081.343 |
| Completion rule | 92.00% | 47 | 2 | 23.50% | 4,121.013 |
| Always recheck | 91.00% | 47 | 6 | 100.00% | 5,521.676 |
| Keep initial | 80.75% | 0 | 0 | 0.00% | 3,005.635 |

Counts are question-weighted equivalents, with 400 independent development
questions represented by 401 rows. Selected and frozen classifiers both have
development macro-F1 0.55872394 and log loss 0.39161267. Action-only baselines
have no three-class probability predictions, so those metrics are N/A.

The selected artifact and frozen artifact produce bitwise identical probabilities
on source training. The threshold change adds 222 development recheck rows
across 221 questions: 221 NO_CHANGE rows and one additional HURT (example 139).
It removes no old rechecks and adds no helpful fixes. This explains the -0.25
percentage-point accuracy change and higher token usage. It does not establish
which future threshold method will generalize better.

Independent read-only verification reproduced all candidate rankings, CV metrics,
threshold decisions, development metrics, and fold-group isolation from saved
predictions. The artifact retains the original 19 features, schema fingerprint,
class order, and artifact format. integrity.json confirms 210 original project
files unchanged and no new files outside outputs/accuracy_experiments/.

The frozen gate historically used this development set, and development was
previously inspected. This comparison is descriptive, not fresh unbiased
confirmation. Source-test and target-test outcomes were not loaded by the
experiment or its numerical verification. No new dataset, downloads, LLM calls,
Streamlit edits, application edits, or deployment occurred.
