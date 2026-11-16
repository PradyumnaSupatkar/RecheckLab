"""Read-only inputs; writes new supplemental diagnostics once."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent; ROOT=P.parents[2]
sys.path.insert(0,str(ROOT))
from rechecklab.study_protocol import normalize_number
f=pd.read_csv(P/'CORRECTION_CASES.csv',keep_default_na=False)
rows=[]; checks=[]
for part,g in f.groupby('partition',sort=False):
    groups=g.groupby('group_id')
    table=groups.apply(lambda x: pd.Series({
        'help':x.original_label.eq('HELP').mean(),
        'length_help':(x.original_label.eq('HELP')&x.base_done_reason.eq('length')).mean(),
        'hurt':x.original_label.eq('HURT').mean(),
        'length_hurt':(x.original_label.eq('HURT')&x.intervention_done_reason.eq('length')).mean(),
    }),include_groups=False)
    a=table.to_numpy();rng=np.random.default_rng(20261116);boot=[]
    for _ in range(2000):
        b=a[rng.integers(0,len(a),len(a))].sum(axis=0)
        if b[0]:boot.append(100*b[1]/b[0])
    rows.append({'partition':part,'length_HELP_pct':100*a[:,1].sum()/a[:,0].sum(),
     'length_HELP_ci_low_pct':np.quantile(boot,.025),'length_HELP_ci_high_pct':np.quantile(boot,.975),
     'revision_length_HURT_pct':100*a[:,3].sum()/a[:,2].sum(),
     'unidentified_content_net_lower_pp':-100*a[:,2].mean(),'unidentified_content_net_upper_pp':100*a[:,0].mean()})
for _,r in f[f.dataset.isin(['commonsenseqa','gsm8k'])].iterrows():
    for phase,col,grade in [('initial','initial_extracted_answer','original_base_correct'),('revision','revised_extracted_answer','original_final_correct')]:
        answer=str(r[col]);key=str(r.published_gold_answer)
        if r.dataset=='commonsenseqa':
            valid=answer in list('ABCDE'); agrees=answer==key
        else:
            norm=normalize_number(answer);gold=normalize_number(key)
            valid=norm is not None and gold is not None;agrees=norm==gold
        # Compare only already-published answer strings in supported syntax.
        # This is not extraction from transcripts and applies no grade changes.
        checks.append({'partition':r.partition,'example_id':r.example_id,'rollout':r.rollout,'phase':phase,
                       'supported_answer_syntax':valid,'published_grade':r[grade],
                       'key_consistent_if_supported':bool(agrees)==bool(r[grade]) if valid else '',
                       'alternative_grade_applied':False})
for name,data in [('UNCERTAINTY.csv',rows),('KEY_CONSISTENCY.csv',checks)]:
    with (P/name).open('x',encoding='utf8',newline='') as h:pd.DataFrame(data).to_csv(h,index=False)
original=json.loads((P/'ORIGINAL_SHA256.json').read_text())
changed=[p for p,digest in original.items() if not (ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest]
assert not changed,changed
with (P/'PRESERVATION_CHECK.json').open('x',encoding='utf8') as h:json.dump({'original_files_checked':len(original),'changed_or_missing':changed},h,indent=2)
print(pd.DataFrame(rows).to_string(index=False))
c=pd.DataFrame(checks);print(c.groupby(['partition','supported_answer_syntax']).size().to_string());print('Supported key mismatches:',len(c[c.key_consistent_if_supported.eq(False)]));print('Preserved originals:',len(original))
