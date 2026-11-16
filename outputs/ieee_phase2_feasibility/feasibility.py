"""Bounded residual feasibility; only source train/dev outcomes are opened.
Run: .venv/Scripts/python.exe -B outputs/ieee_phase2_feasibility/feasibility.py RUN_NAME
Refuses an existing run directory. No production writes, inference or downloads.
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json, hashlib, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rechecklab.features import extract_features, FEATURE_NAMES
from rechecklab.data import validate_pairs, assert_disjoint
from rechecklab.metrics import question_weights, policy_metrics
from rechecklab.predictor import load_artifact, decide_recheck

PENALTY = .01  # one accuracy-probability point per 1,000 extra tokens
ALPHA = 100.0
MARGINS = [0., .01, .025, .05, .1, .2, 1000000.]
SEED = 46

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def snapshot():
    out = {}
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ('.venv', '.git', '__pycache__', '.pytest_cache')]
        for name in names:
            p = Path(folder) / name
            if Path(__file__).parent in p.parents:
                continue
            out[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out

def base(f):
    return f.base_done_reason.eq('length').to_numpy()

def delta(f):
    return f.final_correct.to_numpy(float) - f.base_correct.to_numpy(float)

def masks(f):
    present = f.base_answer.astype(str).str.strip().ne('').to_numpy()
    # Sufficient syntax check only; other mathematical strings remain unassessed.
    numeric = f.base_answer.astype(str).str.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[-+]?\d+)?').to_numpy()
    stop = f.base_done_reason.eq('stop').to_numpy()
    return {'all': np.ones(len(f), bool), 'completed_present_proxy': stop & present,
            'completed_numeric_syntax': stop & numeric,
            'incomplete_length': base(f), 'missing_answer': ~present,
            'nonempty_syntax_unassessed': present & ~numeric,
            'unknown_completion': f.base_done_reason.eq('unknown').to_numpy(),
            'present_initially_wrong_OUTCOME_DIAGNOSTIC': present & ~f.base_correct.to_numpy(bool),
            'numeric_initially_wrong_OUTCOME_DIAGNOSTIC': numeric & ~f.base_correct.to_numpy(bool)}

def opportunity(f, partition):
    rows=[]
    for group, m in masks(f).items():
        if not m.any(): continue
        s=f.loc[m]; w=question_weights(s); n=w.sum(); h=base(s); d=delta(s)
        g=(1-2*h.astype(int))*d; c=s.action_actual_tokens.to_numpy(float)
        oracle=np.where(g>0, ~h, h)  # preserve baseline on neutral transitions
        minimal=s.label.eq('HELP').to_numpy()
        pm=policy_metrics(s,h); om=policy_metrics(s,oracle); mm=policy_metrics(s,minimal)
        rows.append(dict(partition=partition,group=group,questions=int(s.group_id.nunique()),rows=len(s),
            HELP=float(w@(d==1)),HURT=float(w@(d==-1)),NO_CHANGE=float(w@(d==0)),
            initial_wrong=float(w@(~s.base_correct.to_numpy(bool))),
            helpful_add_rechecks=float(w@((~h)&(d==1))),harmful_add_rechecks=float(w@((~h)&(d==-1))),
            helpful_skip_overrides=float(w@(h&(d==-1))),harmful_skip_overrides=float(w@(h&(d==1))),
            beneficial_switches=float(w@(g>0)),accuracy_harming_switches=float(w@(g<0)),
            completion_accuracy_pct=pm['accuracy_pct'],oracle_accuracy_pct=om['accuracy_pct'],
            oracle_accuracy_headroom_pp=om['accuracy_pct']-pm['accuracy_pct'],
            blanket_switch_net_accuracy_pp=float(100*np.average(g,weights=w)),
            completion_avg_tokens=pm['average_total_tokens'],accuracy_oracle_avg_tokens=om['average_total_tokens'],
            minimal_cost_accuracy_oracle_avg_tokens=mm['average_total_tokens'],
            added_help_tokens_total=float(w@((~h)*(d==1)*c)),
            saved_hurt_tokens_total=float(w@(h*(d==-1)*c)),
            net_accuracy_oracle_extra_tokens_per_question=float(np.average((oracle.astype(int)-h)*c,weights=w))))
    return rows

def fit(f):
    x=extract_features(f); w=question_weights(f); h=base(f)
    scaler=StandardScaler().fit(x,sample_weight=w)
    model=Ridge(alpha=ALPHA).fit(scaler.transform(x),(1-2*h.astype(int))*delta(f),sample_weight=w)
    costs={str(v):float(np.average(f.action_actual_tokens.to_numpy(float)[h==v],weights=w[h==v]))
           for v in [False,True] if (h==v).any()}
    return scaler,model,costs

def scores(model,f):
    scaler,ridge,costs=model; h=base(f)
    gain=ridge.predict(scaler.transform(extract_features(f)))
    cost=np.array([costs[str(v)] for v in h])
    return gain-PENALTY*(1-2*h.astype(int))*cost/1000,gain,cost

def decide(f,score,margin): return np.logical_xor(base(f),score>margin)

def select(f,score):
    h=base(f); w=question_weights(f); d=delta(f); c=f.action_actual_tokens.to_numpy(float)
    candidates=[]
    for t in MARGINS:
        a=decide(f,score,t); change=a.astype(int)-h.astype(int)
        acc=float(np.average(change*d,weights=w)); harms=float(w@(change*(d==-1)))
        utility=float(np.average(change*(d-PENALTY*c/1000),weights=w))
        # Empirical conservatism on inner validation, not a probabilistic guarantee.
        if acc>=-1e-12 and harms<=1e-12:
            candidates.append((round(utility,12),-float(np.average(a*c,weights=w)),t))
    return max(candidates)[2]

def split(f,k,seed):
    return list(StratifiedGroupKFold(k,shuffle=True,random_state=seed).split(f,f.label,groups=f.group_id))

def paired(f, a, b):
    c=f.action_actual_tokens.to_numpy(float)
    q=pd.DataFrame({'g':f.group_id.to_numpy(),'acc':(a.astype(int)-b.astype(int))*delta(f),
                    'tokens':(a.astype(int)-b.astype(int))*c}) .groupby('g').mean()
    rng=np.random.default_rng(SEED); idx=rng.integers(0,len(q),size=(2000,len(q)))
    av=q.acc.to_numpy()[idx].mean(1)*100; tv=q.tokens.to_numpy()[idx].mean(1)
    return [float(v) for v in np.quantile(av,[.025,.975])], [float(v) for v in np.quantile(tv,[.025,.975])]

def evaluate(f,a,partition,status,policy,scope):
    h=base(f); w=question_weights(f); d=delta(f); change=a.astype(int)-h.astype(int)
    m=policy_metrics(f,a); bm=policy_metrics(f,h); ci,tc=paired(f,a,h)
    beneficial=(1-2*h.astype(int))*d>0
    return dict(partition=partition,evaluation_status=status,scope=scope,policy=policy,
        **m,accuracy_difference_vs_completion_pp=m['accuracy_pct']-bm['accuracy_pct'],
        token_difference_vs_completion=m['average_total_tokens']-bm['average_total_tokens'],
        token_savings_vs_completion_pct=100*(1-m['average_total_tokens']/bm['average_total_tokens']),
        helpful_corrections=float(w@(a*(d==1))),harmful_revisions=float(w@(a*(d==-1))),
        additional_helpful_corrections=float(w@(change*(d==1))),additional_harmful_revisions=float(w@(change*(d==-1))),
        beneficial_overrides_taken=float(w@((a!=h)&beneficial)),
        helpful_overrides_missed=float(w@((a==h)&beneficial)),
        accuracy_harming_overrides=float(w@((a!=h)&((1-2*h.astype(int))*d<0))),
        override_pct=100*float(np.average(a!=h,weights=w)),
        accuracy_difference_ci_low_pp=ci[0],accuracy_difference_ci_high_pp=ci[1],
        token_difference_ci_low=tc[0],token_difference_ci_high=tc[1])

def policy_rows(f,choices,partition):
    rows=[]
    for name,a,status in choices:
        for scope in ['all','completed_present_proxy','completed_numeric_syntax']:
            m=masks(f)[scope]
            if m.any(): rows.append(evaluate(f.loc[m],a[m],partition,status,name,scope))
    return rows

def matched(f,gain):
    # Batch quota from initial-only completion decisions, with group-level ties.
    q=pd.DataFrame({'g':f.group_id.to_numpy(),'value':(1-2*base(f).astype(int))*gain,'h':base(f)})
    q=q.groupby('g').mean().reset_index().sort_values(['value','g'],ascending=[False,True])
    chosen=set(q.head(int(q.h.sum())).g)
    return f.group_id.isin(chosen).to_numpy()

def main():
    name=sys.argv[1] if len(sys.argv)>1 else 'run_2026-11-16_01'
    if Path(name).name!=name: raise ValueError('Run name must be a single directory name')
    run=Path(__file__).parent/name
    before=snapshot(); run.mkdir(exist_ok=False)
    dump(run/'original_files_before.json',before)
    plan={'seed':SEED,'ridge_alpha':ALPHA,'penalty_per_1000_tokens':PENALTY,'threshold_grid':MARGINS,
        'outer_folds':5,'inner_folds':3,'bootstrap_repeats':2000,'features':FEATURE_NAMES,
        'selection':'inner utility, require empirical accuracy delta >=0 and harm delta <=0; token tie-break then larger margin',
        'labels_opened':['source_train','source_development'],
        'validity':'nonempty publisher answer is presence proxy, not verified mathematical validity; numeric syntax sufficient only',
        'frozen_baseline_warning':'source training is in-sample; development selected frozen model historically',
        'safety':'margin and empirical constraints are not confidence bounds',
        'candidate_count':1,'final_threshold':'training outer-OOF tuning; no unbiased claim for optimized tuning surface'}
    dump(run/'analysis_plan.json',plan)
    train=validate_pairs(pd.read_parquet(ROOT/'outputs/v0.2/data/prepared/source_train.parquet'))
    opp=opportunity(train,'source_train_exploratory')
    pd.DataFrame(opp).to_csv(run/'TRAINING_OPPORTUNITY.csv',index=False)
    with threadpool_limits(limits=1):
        outer=split(train,5,SEED); oof=np.zeros(len(train)); gains=np.zeros(len(train)); actions=np.zeros(len(train),bool)
        foldids=np.zeros(len(train),int); folds=[]
        for j,(tr,va) in enumerate(outer):
            a=train.iloc[tr]; b=train.iloc[va]; assert_disjoint(a,b)
            inner=np.zeros(len(a)); inner_groups=[]
            for it,iv in split(a,3,SEED+100+j):
                assert_disjoint(a.iloc[it],a.iloc[iv]); inner[iv]=scores(fit(a.iloc[it]),a.iloc[iv])[0]
                inner_groups.append({'fit':sorted(a.iloc[it].group_id.unique()),'validation':sorted(a.iloc[iv].group_id.unique())})
            threshold=select(a,inner); s,g,c=scores(fit(a),b)
            oof[va]=s;gains[va]=g;actions[va]=decide(b,s,threshold);foldids[va]=j
            folds.append({'fold':j,'threshold':threshold,'fit_groups':sorted(a.group_id.unique()),
                          'validation_groups':sorted(b.group_id.unique()),'inner_groups':inner_groups,
                          'validation_beneficial_switches':int(((1-2*base(b).astype(int))*delta(b)>0).sum())})
        final_t=select(train,oof); final_model=fit(train)
        dump(run/'frozen_candidate.json',{'threshold':final_t,'alpha':ALPHA,'penalty':PENALTY,
            'cost_estimates':final_model[2],'feature_names':FEATURE_NAMES,
            'coefficients':final_model[1].coef_.tolist(),'intercept':float(final_model[1].intercept_),
            'scaler_mean':final_model[0].mean_.tolist(),'scaler_scale':final_model[0].scale_.tolist(),
            'frozen_before_development_load':True})
        dump(run/'folds.json',folds)
        pred=train[['group_id','example_id','label']].copy();pred['fold']=foldids;pred['score']=oof;pred['gain_prediction']=gains;pred['decision']=actions
        pred.to_csv(run/'training_oof_predictions.csv',index=False)
        # Associations are exploratory only; no feature or direction is selected from them.
        assoc=[];x=extract_features(train);y=(1-2*base(train).astype(int))*delta(train)>0
        for group in ['completed_present_proxy','incomplete_length']:
            m=masks(train)[group]
            for feature in FEATURE_NAMES:
                values=x.loc[m,feature].to_numpy(); yy=y[m]
                if len(np.unique(values))<2 or len(np.unique(yy))<2:continue
                assoc.append({'group':group,'feature':feature,'positive_switches':int(yy.sum()),
                    'raw_direction_auc_exploratory':roc_auc_score(yy,values),
                    'mean_beneficial':float(values[yy].mean()),'mean_other':float(values[~yy].mean())})
            if len(np.unique(y[m]))==2:
                assoc.append({'group':group,'feature':'NESTED_OOF_RESIDUAL_SCORE','positive_switches':int(y[m].sum()),
                    'raw_direction_auc_exploratory':roc_auc_score(y[m],oof[m])})
        pd.DataFrame(assoc).to_csv(run/'feature_associations.csv',index=False)
        frozen=load_artifact(ROOT/'outputs/v0.2/models/source_gate.joblib')
        fg=lambda f:decide_recheck(frozen['predictor'].probabilities(f),frozen['threshold'],frozen['harm_weight'])
        choices=[('Completion',base(train),'fixed heuristic'),('Frozen ML',fg(train),'IN_SAMPLE pretrained on these questions'),
            ('Residual',actions,'nested grouped OOF; exploratory design; overlapping fits'),
            ('Residual matched rate',matched(train,gains),'OOF initial-only batch quota diagnostic'),
            ('Oracle',train.label.eq('HELP').to_numpy(),'unattainable hindsight HELP-only')]
        rows=policy_rows(train,choices,'source_training')
        # First source-development outcome access: all candidate choices above frozen.
        dev=validate_pairs(pd.read_parquet(ROOT/'outputs/v0.2/data/prepared/source_development.parquet'))
        assert_disjoint(train,dev);s,g,c=scores(final_model,dev)
        a=decide(dev,s,final_t)
        choices=[('Completion',base(dev),'descriptive development confirmation'),
            ('Frozen ML',fg(dev),'retrospective development-selected baseline'),
            ('Residual',a,'frozen candidate; previously exposed development descriptive only'),
            ('Residual matched rate',matched(dev,g),'initial-only batch quota diagnostic; not selected'),
            ('Oracle',dev.label.eq('HELP').to_numpy(),'unattainable hindsight HELP-only')]
        rows+=policy_rows(dev,choices,'source_development')
        pd.DataFrame(rows).to_csv(run/'POLICY_COMPARISON.csv',index=False)
        pred=dev[['group_id','example_id','label']].copy();pred['score']=s;pred['gain_prediction']=g;pred['estimated_cost']=c
        for name,mask,_ in choices:pred[name]=mask
        pred.to_csv(run/'development_predictions.csv',index=False)
        opp+=opportunity(dev,'source_development_descriptive')
        pd.DataFrame(opp).to_csv(run/'OPPORTUNITY.csv',index=False)
    after=snapshot(); changed=[p for p in before if after.get(p)!=before[p]];added=sorted(set(after)-set(before))
    dump(run/'integrity.json',{'original_files':len(before),'changed_or_removed':changed,'new_outside_phase2':added})
    assert not changed and not added
    print(json.dumps({'run':str(run),'threshold':final_t,'outer_thresholds':[r['threshold'] for r in folds],
        'results':[r for r in rows if r['scope']=='all'],'protected_files':len(before)},indent=2))

if __name__=='__main__':main()
