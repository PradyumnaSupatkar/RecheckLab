"""CPU-only measurement audit. Run with --output a NEW directory to reproduce.
No regrading, fitting, backend access, or writes outside the output directory.
"""
from pathlib import Path
import argparse, hashlib, json, re, sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from rechecklab.data import question_hash, transition_label
from rechecklab.metrics import policy_metrics

NOTES = {
 '253': 'Unicode superscript in asserted final polynomial; expanded coefficients independently agree. Representation suspect, no published key.',
 '572': 'Empty extracted answer despite explicit multiline FINAL followed by fraction 1/4; logarithm derivation agrees. Extraction suspect.',
 '1203': 'Final ordered pair (3,-1) enclosed in answer tags; extracted string retains tags and loses comma. Both equations check.',
 '1226': 'Repeated Final answer prefix retained in extraction; asserted pair (0.5,0.6) satisfies both equations.',
 '1355': 'Explicit final polynomial uses Unicode superscripts. Expansion agrees; no reasoning trace to assess.',
 '1373': 'LaTeX text-wrapper closing brace retained before fraction; discriminant argument supports 29/4.',
 '1494': 'Repeated Final answer prefix retained; coefficient addition supports final polynomial.'}

def write_new(path, text):
    with path.open('x', encoding='utf8', newline='') as f: f.write(text)

def key_maps():
    maps = {}
    for name, split in [('commonsenseqa','validation'), ('gsm8k','test')]:
        files = list((ROOT/'outputs/live_study/v0.3.0/raw'/name).rglob(split+'.parquet'))
        mapping = {}
        if files:
            for _, r in pd.read_parquet(files[0]).iterrows():
                if name == 'commonsenseqa':
                    choices = '\n'.join(f'{a}. {b}' for a,b in zip(r.choices['label'],r.choices['text']))
                    query = r.question+'\n\nChoices:\n'+choices+'\n\nAnswer with only the correct option letter.'
                    key = r.answerKey
                else:
                    query = r.question; key = r.answer.split('####')[-1].strip()
                h = question_hash(query)
                assert h not in mapping or mapping[h] == key
                mapping[h] = key
        maps[name] = mapping
    return maps

def category(r, partition):
    if r.label == 'NO_CHANGE':
        return 'PUBLISHED_STABLE_CORRECT' if r.base_correct else 'PUBLISHED_STABLE_WRONG'
    if r.label == 'HELP':
        if partition == 'source_train' and str(r.example_id) in NOTES:
            return 'REVIEWED_REPRESENTATION_RECOVERY_SUSPECT'
        if r.base_done_reason == 'length': return 'TRUNCATION_ASSOCIATED_HELP_UNRESOLVED'
        if not str(r.base_answer).strip(): return 'MISSING_ANSWER_HELP_UNRESOLVED'
        return 'HELP_CAUSE_UNRESOLVED'
    if r.intervention_done_reason == 'length': return 'TRUNCATION_ASSOCIATED_HURT_UNRESOLVED'
    return 'HURT_CAUSE_UNRESOLVED'

def strict_proxy(r):
    lines = str(r.base_response).strip().splitlines()
    return r.base_done_reason == 'stop' and bool(str(r.base_answer).strip()) and bool(lines) and bool(re.search(r'\bFINAL(?:\s+ANSWER)?\s*:\s*\S',lines[-1],re.I))

def main(out):
    if out != HERE:
        out.mkdir(parents=True,exist_ok=False)
        write_new(out/'FROZEN_AUDIT_PROTOCOL.md',(HERE/'FROZEN_AUDIT_PROTOCOL.md').read_text(encoding='utf8'))
    assert (out/'FROZEN_AUDIT_PROTOCOL.md').is_file()
    for name in ['CORRECTION_CASES.csv','SUMMARY.csv','CATEGORY_COUNTS.csv','SUBGROUP_POLICIES.csv','SOURCE_TRAIN_REVIEW.json','RUN_MANIFEST.json']:
        if (out/name).exists(): raise FileExistsError(out/name)
    keys = key_maps(); cases=[]; summaries=[]; cats=[]; policies=[]; manual=[]; inputs={}
    paths = [(name, ROOT/'outputs/v0.2/data/prepared'/f'{name}.parquet') for name in ['source_train','source_development','source_test','target_adaptation_pool','target_test']]
    paths += [(name+'_replay',ROOT/'data/raw'/f'{name}.parquet') for name in ['math500','gsm8k']]
    for part,path in paths:
        inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        f=pd.read_parquet(path); f=f[f.action.eq('active_verify')].copy().reset_index(drop=True)
        f['group_id']=f['query'].map(question_hash)
        labels=[transition_label(bool(a),bool(b)) for a,b in zip(f.base_correct,f.final_correct)]
        if 'label' in f: assert list(f.label)==labels
        f['label']=labels
        assert np.array_equal(f.helpful_fix.astype(bool),f.label.eq('HELP'))
        assert np.array_equal(f.harmful_flip.astype(bool),f.label.eq('HURT'))
        assert np.array_equal(f.total_actual_tokens, f.base_actual_tokens+f.action_actual_tokens)
        # Canonical duplicate questions can have different initial transcripts.
        # Preserve that dependence without pretending they are one base attempt.
        differing_initial_transcripts=int((f.groupby('group_id').base_response.nunique(dropna=False)>1).sum())
        for c in ['base_answer','base_correct','base_done_reason']:
            assert f.groupby('group_id')[c].nunique(dropna=False).max()==1
        w=1/f.groupby('group_id').group_id.transform('size')
        n=f.group_id.nunique(); f['audit_category']=[category(r,part) for _,r in f.iterrows()]
        f['strict_proxy']=[strict_proxy(r) for _,r in f.iterrows()]
        f['stop_nonempty']=f.base_done_reason.eq('stop') & f.base_answer.astype(str).str.strip().ne('')
        delta=(f.final_correct.astype(float)-f.base_correct.astype(float)).groupby(f.group_id).mean().to_numpy()
        rng=np.random.default_rng(20261116)
        boot=np.array([rng.choice(delta,len(delta),replace=True).mean()*100 for _ in range(2000)])
        s={'partition':part,'status':'training descriptive' if part=='source_train' else 'exposed/descriptive; not independent confirmation','questions':n,'rollout_rows':len(f),'canonical_groups_with_different_initial_transcripts':differing_initial_transcripts,
           'initial_accuracy_pct':100*np.sum(w*f.base_correct)/n,'always_accuracy_pct':100*np.sum(w*f.final_correct)/n,
           'gain_pp':100*delta.mean(),'gain_ci_low_pp':np.quantile(boot,.025),'gain_ci_high_pp':np.quantile(boot,.975)}
        for label in ['HELP','HURT','NO_CHANGE']:
            s[label+'_rows']=int(f.label.eq(label).sum());s[label+'_question_equivalents']=float(w[f.label.eq(label)].sum())
        for c in ['base_finalizer_used','finalizer_used']:
            s[c+'_question_equivalents']=float(w[f[c].astype(bool)].sum())
        s['HELP_length_question_equivalents']=float(w[f.label.eq('HELP')&f.base_done_reason.eq('length')].sum())
        summaries.append(s)
        for c,g in f.groupby('audit_category'):
            mass=float(w.loc[g.index].sum())
            cats.append({'partition':part,'category':c,'rows':len(g),'question_equivalents':mass,
                         'fraction_of_HELP':mass/s['HELP_question_equivalents'] if g.label.eq('HELP').all() and s['HELP_question_equivalents'] else None})
        for subgroup,mask in [('all',np.ones(len(f),bool)),('length',f.base_done_reason.eq('length')),('stop_nonempty',f.stop_nonempty),('strict_FINAL_proxy',f.strict_proxy)]:
            g=f.loc[mask].copy()
            if g.empty: continue
            decisions={'never':np.zeros(len(g),bool),'always':np.ones(len(g),bool),'completion':g.base_done_reason.eq('length').to_numpy()}
            for p,d in decisions.items():
                policies.append({'partition':part,'subgroup':subgroup,'policy':p,**policy_metrics(g,d)})
        name='commonsenseqa' if part.startswith('target') else 'gsm8k' if part=='gsm8k_replay' else 'math'
        for i,r in f.iterrows():
            note=NOTES.get(str(r.example_id),'') if part=='source_train' else ''
            key=keys.get(name,{}).get(r.group_id,'')
            cases.append({'dataset':name,'partition':part,'example_id':r.example_id,'rollout':r.rollout,'group_id':r.group_id,'question_weight':w[i],
             'source_file':str(path.relative_to(ROOT)),'source_row_after_active_filter':i,'query':r.query,
             'base_done_reason':r.base_done_reason,'intervention_done_reason':r.intervention_done_reason,
             'base_finalizer_used':r.base_finalizer_used,'finalizer_used':r.finalizer_used,
             'initial_extracted_answer':r.base_answer,'revised_extracted_answer':r.final_answer,
             'published_gold_answer':key,'gold_key_status':'exact full-query local benchmark join' if key else 'unavailable/unmatched; revised answer is NOT gold',
             'original_base_correct':r.base_correct,'original_final_correct':r.final_correct,'original_label':r.label,
             'primary_observational_category':r.audit_category,'strict_FINAL_proxy':r.strict_proxy,
             'initial_excerpt':str(r.base_response)[-1600:],'revised_excerpt':str(r.intervention_response)[-1000:],
             'confidence':'high representation suspicion; causal mechanism unverified' if note else 'high recorded-state confidence; semantic interpretation unassessed',
             'ambiguity_notes':note or 'No manual semantic adjudication. Grade transition does not establish reasoning change.',
             'alternative_grade_applied':False})
            if note:
                manual.append({k:r[k] for k in ['example_id','query','base_answer','final_answer','base_response','intervention_response','base_done_reason','base_finalizer_used'] }|{'note':note,'published_gold_answer':None})
    for name,rows in [('CORRECTION_CASES.csv',cases),('SUMMARY.csv',summaries),('CATEGORY_COUNTS.csv',cats),('SUBGROUP_POLICIES.csv',policies)]:
        with (out/name).open('x',encoding='utf8',newline='') as h: pd.DataFrame(rows).to_csv(h,index=False)
    write_new(out/'SOURCE_TRAIN_REVIEW.json',json.dumps(manual,ensure_ascii=False,indent=2,default=str))
    manifest={'inputs':inputs,'protocol_sha256':hashlib.sha256((out/'FROZEN_AUDIT_PROTOCOL.md').read_bytes()).hexdigest(),'bootstrap_seed':20261116,'bootstrap_repeats':2000,'grade_changes':0,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}
    write_new(out/'RUN_MANIFEST.json',json.dumps(manifest,indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE)
    main(parser.parse_args().output.resolve())
