"""Independent arithmetic verification of saved Phase 2 outputs; no fitting/tests/inference."""
import sys, json, hashlib
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
run=Path(__file__).parent/(sys.argv[1] if len(sys.argv)>1 else 'run_2026-11-16_01')
sys.path.insert(0,str(ROOT))
from rechecklab.features import extract_features
from rechecklab.predictor import load_artifact, decide_recheck
from threadpoolctl import threadpool_limits

def check():
    comparisons=pd.read_csv(run/'POLICY_COMPARISON.csv',float_precision='round_trip')
    folds=json.loads((run/'folds.json').read_text())
    cfg=json.loads((run/'frozen_candidate.json').read_text())
    count=0
    for split,filename in [('source_training','source_train'),('source_development','source_development')]:
        f=pd.read_parquet(ROOT/f'outputs/v0.2/data/prepared/{filename}.parquet')
        pred=pd.read_csv(run/('training_oof_predictions.csv' if split=='source_training' else 'development_predictions.csv'),float_precision='round_trip',dtype={'example_id':str})
        assert f.group_id.tolist()==pred.group_id.tolist()
        h=f.base_done_reason.eq('length').to_numpy();d=f.final_correct.astype(float).to_numpy()-f.base_correct.astype(float).to_numpy()
        if split=='source_training':
            seen=[];a=np.zeros(len(f),bool)
            for fold in folds:
                fit=set(fold['fit_groups']);val=set(fold['validation_groups']);assert not fit&val
                for inner in fold['inner_groups']:
                    it=set(inner['fit']);iv=set(inner['validation']);assert not it&iv and it|iv==fit
                seen+=list(val);m=f.group_id.isin(val).to_numpy()
                a[m]=np.logical_xor(h[m],pred.score.to_numpy()[m]>fold['threshold'])
            assert len(seen)==len(set(seen))==f.group_id.nunique()
            assert np.array_equal(a,pred.decision.to_numpy(bool))
        else:
            x=extract_features(f).to_numpy();gain=((x-cfg['scaler_mean'])/cfg['scaler_scale'])@np.array(cfg['coefficients'])+cfg['intercept']
            costs=np.array([cfg['cost_estimates'][str(v)] for v in h]);score=gain-cfg['penalty']*(1-2*h.astype(int))*costs/1000
            assert np.allclose(score,pred.score.to_numpy(),atol=1e-12)
            a=np.logical_xor(h,score>cfg['threshold']);assert np.array_equal(a,pred.Residual.to_numpy(bool))
        bundle=load_artifact(ROOT/'outputs/v0.2/models/source_gate.joblib')
        with threadpool_limits(limits=1):
            frozen=decide_recheck(bundle['predictor'].probabilities(f),bundle['threshold'],bundle['harm_weight'])
        rank=pd.DataFrame({'g':f.group_id,'v':(1-2*h.astype(int))*pred.gain_prediction.to_numpy(),'h':h}).groupby('g').mean().reset_index().sort_values(['v','g'],ascending=[False,True])
        quota=f.group_id.isin(set(rank.head(int(rank.h.sum())).g)).to_numpy()
        decisions={'Completion':h,'Frozen ML':frozen,'Residual':a,'Residual matched rate':quota,'Oracle':d==1}
        for _,r in comparisons[comparisons.partition.eq(split)].iterrows():
            m=np.ones(len(f),bool)
            if r.scope=='completed_present_proxy':m=(f.base_done_reason.eq('stop')&f.base_answer.astype(str).str.strip().ne('')).to_numpy()
            if r.scope=='completed_numeric_syntax':m=(f.base_done_reason.eq('stop')&f.base_answer.astype(str).str.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[-+]?\d+)?')).to_numpy()
            s=f.loc[m];aa=decisions[r.policy][m];hh=h[m];dd=d[m];w=1/s.groupby('group_id').group_id.transform('size').to_numpy();change=aa.astype(int)-hh.astype(int)
            values={'accuracy_pct':100*np.average(s.base_correct.astype(float)+aa*dd,weights=w),
                'average_total_tokens':np.average(s.base_actual_tokens+aa*s.action_actual_tokens,weights=w),
                'recheck_pct':100*np.average(aa,weights=w),'override_pct':100*np.average(aa!=hh,weights=w),
                'helpful_corrections':w@(aa*(dd==1)),'harmful_revisions':w@(aa*(dd==-1)),
                'additional_helpful_corrections':w@(change*(dd==1)), 'additional_harmful_revisions':w@(change*(dd==-1)),
                'accuracy_difference_vs_completion_pp':100*np.average(change*dd,weights=w)}
            for k,v in values.items():assert np.isclose(v,r[k],atol=1e-10),(split,r.policy,r.scope,k,v,r[k]);count+=1
        if split=='source_training':
            details=f[f.base_done_reason.eq('stop')&f.label.eq('HELP')][['example_id','query','base_answer','final_answer','base_response','label']].to_dict('records')
            with (run/'training_completed_HELP_evidence.json').open('x',encoding='utf-8') as out:json.dump(details,out,indent=2,ensure_ascii=True)
    before=json.loads((run/'original_files_before.json').read_text());changed=[p for p,h in before.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed
    result={'numeric_comparisons':count,'original_files_unchanged':len(before),'fold_groups_verified':True,'coefficient_reconstruction_verified':True,'new_fits':0,'test_outcomes_opened':False}
    with (run/'verification.json').open('x') as out:json.dump(result,out,indent=2)
    print(json.dumps(result))

if __name__=='__main__':check()
