"""A compact offline report and one reproducible adaptation figure."""
from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def render_report(root):
    reports = Path(root) / 'reports'
    metadata = json.loads((reports / 'experiment_metadata.json').read_text(encoding='utf-8'))
    manifest = json.loads((reports / 'split_manifest.json').read_text(encoding='utf-8'))
    policies = pd.read_csv(reports / 'policy_results.csv')
    adaptation = pd.read_csv(reports / 'adaptation_summary.csv')
    target = policies.loc[policies.evaluation == 'Target test']
    frozen = target.loc[target.policy == 'Frozen source gate'].iloc[0]
    completion = target.loc[target.policy == 'Completion rule'].iloc[0]
    keep = target.loc[target.policy == 'Keep initial answer'].iloc[0]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axis = plt.subplots(figsize=(8, 4.3), layout='constrained')
    axis.errorbar(adaptation.adaptation_questions, adaptation.accuracy_mean,
        yerr=adaptation.accuracy_std, marker='o', color='#376382', capsize=4,
        label='Threshold adaptation: mean and subset SD')
    if abs(frozen.accuracy_pct - completion.accuracy_pct) < 1e-10:
        axis.axhline(frozen.accuracy_pct, color='#50675e', linestyle='--', label='Frozen gate / completion rule')
    else:
        axis.axhline(frozen.accuracy_pct, color='#50675e', linestyle='--', label='Frozen gate')
        axis.axhline(completion.accuracy_pct, color='#888888', linestyle=':', label='Completion rule')
    axis.axhline(keep.accuracy_pct, color='#aaaaaa', linestyle=':', label='Keep initial')
    axis.set(xlabel='Target adaptation questions (reserved pool)', ylabel='Fixed target-test accuracy (%)',
             title='Does threshold-only adaptation improve the policy?')
    axis.grid(axis='y', alpha=.18)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(reports / 'adaptation_curve.png', dpi=160)
    plt.close(fig)
    metadata['artifact_sha256']['reports/adaptation_curve.png'] = hashlib.sha256(
        (reports / 'adaptation_curve.png').read_bytes()).hexdigest()
    (reports / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    diagnostic = metadata['target_predictor_metrics']
    majority = metadata['candidate_test_diagnostics']['majority_class']['target_test']
    sentence = (f"Frozen gate target accuracy: {frozen.accuracy_pct:.2f}%; completion rule: {completion.accuracy_pct:.2f}%; "
                f"keep initial: {keep.accuracy_pct:.2f}%. Target outcome macro-F1: {diagnostic['macro_f1']:.3f} "
                f"(majority baseline {majority['macro_f1']:.3f}); multiclass log loss: {diagnostic['log_loss']:.3f}.")
    if adaptation.accuracy_mean.max() <= frozen.accuracy_pct:
        sentence += ' No tested label budget improved mean accuracy over the frozen gate.'
    else:
        sentence += ' Some adaptation means exceed the frozen gate on this partition; overlapping subsets and one split do not establish a reliable gain.'
    if frozen.accuracy_pct <= completion.accuracy_pct:
        sentence += ' The learned gate does not beat the completion rule in accuracy.'
    if diagnostic['predicted_class_counts']['NO_CHANGE'] == diagnostic['rollout_rows']:
        sentence += ' The selected predictor assigns NO_CHANGE to every target row; policy thresholding uses probabilities rather than this argmax class.'
    source_count = sum(manifest['partitions'][name]['questions'] for name in ('source_train', 'source_development', 'source_test'))
    target_count = sum(manifest['partitions'][name]['questions'] for name in ('target_adaptation_pool', 'target_test'))
    details = html.escape(json.dumps(metadata, indent=2))
    image = 'data:image/png;base64,' + base64.b64encode((reports / 'adaptation_curve.png').read_bytes()).decode()
    table = target[['policy', 'accuracy_pct', 'harmful_flip_pct_of_all_questions', 'recheck_pct', 'average_total_tokens']].round(2).to_html(index=False, border=0)
    limitations = ''.join('<li>' + html.escape(item) + '</li>' for item in metadata['limitations'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RecheckLab v0.2 research report</title><style>body{{font:16px/1.6 system-ui;color:#263445;max-width:960px;margin:32px auto;padding:0 24px}}h1{{font-size:28px}}table{{border-collapse:collapse;font-size:14px;width:100%}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}img{{width:100%}}pre{{white-space:pre-wrap;font-size:12px}}summary{{cursor:pointer}}</style>
<h1>RecheckLab v0.2: one bounded LLM recheck</h1><p>{html.escape(metadata['provenance'])}</p>
<p>Predict HELP (wrong to correct), HURT (correct to wrong), or NO_CHANGE (correctness unchanged). A separate policy chooses whether to keep the initial answer or use the recorded revision. Skipping never certifies correctness.</p>
<p>{source_count:,} source maths questions; {target_count:,} target questions. {len(metadata['features'])} initial-only features. All rollouts stay with their question.</p>
<p>{html.escape(sentence)}</p>{table}<p>Token use and savings are replay estimates; they exclude gate overhead and do not establish compute-optimal decisions.</p>
<img src="{image}" alt="Threshold-only adaptation means with standard deviation across overlapping subsets">
<p>The classifier and probability estimates stay frozen. The 200-question budget and first subset are fixed for the displayed adapted gate; target test labels never choose settings. Error bars show subset SD, not confidence intervals.</p>
<details><summary>Model diagnostics, safeguards and limitations</summary><ul>{limitations}</ul><pre>{details}</pre></details>
<p>Data and authors: <a href="https://huggingface.co/datasets/Sajib-006/sevra-recovery-outcomes">SEVRA Recovery Outcomes</a>, pinned revision 10411137fcaa02b3bd549a2285b43ffe315417f6; Sajib Acharjee Dip, Dawei Zhou and Liqing Zhang. The unchanged original dataset card and underlying benchmark terms are included in data/raw.</p></html>'''
    (reports / 'Progress_Report.html').write_text(page, encoding='utf-8')
    note = f'''# RecheckLab v0.2 progress

Date: 4 October 2026

The offline demo predicts the outcome of one specified recheck and separately chooses whether to spend the extra call. It replays published responses; no new LLM inference was executed. Skip recheck keeps an answer that may still be wrong.

{source_count:,} source maths questions and {target_count:,} target CommonsenseQA questions. Source-only model selection chose {metadata['model']}. Majority, unbalanced and balanced logistic/boosting models were compared. Metrics and uncertainty give each question equal weight across its rollouts.

{sentence}

Adaptation changes only the decision threshold using the reserved target pool. It does not calibrate probabilities. Target test labels are excluded from fitting and setting thresholds.

The two-tab Streamlit app uses actual recorded initial tokens/completion/finalizer values, validates matching model and results, and reveals the revision/outcome after prediction. Raw data and v0.1 results remain unchanged. See the root GUIDE_PROGRESS.md for completed verification and audit details.

Limits: one solver, one transfer direction and split; rare harmful source outcomes; inherited labels; incomplete public generation metadata; uncalibrated probabilities; token savings are replay estimates. No established advantage over the completion rule or reliable adaptation gain is assumed.
'''
    (reports / 'GUIDE_PROGRESS.md').write_text(note, encoding='utf-8')
    return sentence
