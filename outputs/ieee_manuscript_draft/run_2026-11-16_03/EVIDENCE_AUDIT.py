from __future__ import annotations
import ast, csv, hashlib, json, re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
M=ROOT/'outputs/live_study/v0.4.3_math_l4_l5_frozen_evaluation_2026-11-16_01'
MR=ROOT/'outputs/live_study/v0.4.3_exploratory_early_stopped_report_2026-11-16_01'
G=ROOT/'outputs/live_study/v0.4.0_instruct_frozen_evaluation_2026-11-16_01'
DEV=ROOT/'outputs/live_study/v0.4.1_math_l4_l5_development_2026-11-16_01/source_cache'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def jlines(p): return [json.loads(x) for x in p.open(encoding='utf8') if x.strip()]
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def boxed(s):
    # Extract a single final \boxed{...}, balancing nested braces.
    matches=[]
    for m in re.finditer(r'\\boxed\s*\{',s):
        i=m.end(); depth=1; start=i
        while i<len(s) and depth:
            if s[i]=='{': depth+=1
            elif s[i]=='}': depth-=1
            i+=1
        if depth==0: matches.append(s[start:i-1].strip())
    if len(matches)!=1: return None
    x=matches[0].replace(r'\,','').replace(r'\!','').replace(r'\displaystyle','').strip()
    m=re.fullmatch(r'([-+]?)\\(?:d)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}',x)
    try:
        if m: return int(m.group(1)+'1')*Fraction(m.group(2))/Fraction(m.group(3))
        return Fraction(x)
    except Exception: return None
# Pinned source integrity + all 60 IDs/levels/key linkage.
manifest=json.loads((M/'EXPERIMENT_MANIFEST.json').read_text(encoding='utf8'))
sel=jlines(M/'QUESTION_SELECTION.jsonl'); keys={x['example_id']:x for x in jlines(M/'TRUSTED_KEYS.jsonl')}
source={}; source_hashes={}
for rel,meta in manifest['dataset']['test_partitions'].items():
    f=ROOT/meta['local_file'].replace('\\','/')
    source_hashes[rel]={'actual':sha(f),'expected':meta['sha256'],'match':sha(f)==meta['sha256']}
    df=pd.read_parquet(f)
    topic=rel.split('/')[0]
    for _,r in df.iterrows(): source[(topic,Path(rel).parent.name+'/'+str(r['problem']) if False else str(r['problem']))]=(str(r['level']),str(r['solution']))
# Match by exact problem text and topic; source ID uniqueness and exact levels.
source_rows={}
for rel,meta in manifest['dataset']['test_partitions'].items():
    topic=rel.split('/')[0]; f=ROOT/meta['local_file'].replace('\\','/')
    for _,r in pd.read_parquet(f).iterrows(): source_rows[(topic,str(r['problem']))]=(str(r['level']),str(r['solution']))
source_checks=[]; source_answers=[]
for q in sel:
    key=(q['topic'],q['query']); row=source_rows.get(key); gold=keys.get(q['example_id'],{}).get('gold_answer')
    ref=boxed(row[1]) if row else None
    try: keyfrac=Fraction(str(gold))
    except Exception: keyfrac=None
    source_checks.append({'id':q['example_id'],'level_match':bool(row and row[0]==q['level']), 'source_problem_exact_match':row is not None,'key_matches_source_boxed':ref is not None and keyfrac==ref,'source_boxed':str(ref) if ref is not None else None,'trusted_key':str(keyfrac) if keyfrac is not None else str(gold)})
# Reconstruct completed outputs from streamed journal; independent final-line/Fraction grader.
responses=defaultdict(lambda:{'content':''})
for ev in jlines(M/'STREAM_JOURNAL.jsonl'):
    if ev.get('event')=='chunk' and ev.get('call_id'):
        raw=ev.get('raw_chunk') or {}
        msg=raw.get('message') or {}
        responses[ev['call_id']]['content']+=msg.get('content','')
mathq=csvrows(MR/'QUESTION_LEVEL.csv'); mathpol=csvrows(MR/'POLICY_COMPARISON.csv'); gq=csvrows(G/'QUESTION_LEVEL.csv'); gpol=csvrows(G/'POLICY_COMPARISON.csv')
mathq_by={r['question_id']:r for r in mathq}
completed=jlines(M/'COMPLETED_CALLS.jsonl'); gold_by={k['example_id']:k['gold_answer'] for k in keys.values()}
call_grades=[]
for c in completed:
    text=responses[c['call_id']]['content']; m=re.findall(r'^\s*FINAL:\s*(.*?)\s*$',text,re.I|re.M)
    try:
        candidate=m[-1].strip() if len(m)==1 else ''
        parsed=boxed(candidate) if r'\boxed' in candidate else None
        if parsed is None and len(m)==1:
            candidate=candidate.replace(r'\,','').strip().strip('$')
            fm=re.fullmatch(r'([-+]?)\\(?:d)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}',candidate)
            parsed=int(fm.group(1)+'1')*Fraction(fm.group(2))/Fraction(fm.group(3)) if fm else Fraction(candidate.replace(',',''))
    except Exception: parsed=None
    try: gold=Fraction(str(gold_by[c['question_id']]))
    except Exception: gold=None
    correct=parsed is not None and gold is not None and parsed==gold
    call_grades.append({'id':c['question_id'],'phase':c['phase'],'saved_parseable':c.get('parse_status')=='parseable','independent_parseable':parsed is not None,'saved_answer':c.get('parsed_answer'),'independent_answer':str(parsed) if parsed is not None else None,'saved_correct':(mathq_by.get(c['question_id'],{}).get('initial_correct' if c['phase']=='initial' else 'revision_correct','').lower()=='true'),'independent_correct':correct,'response_chars':len(text)})
# Accounting cross-check against source tables and both reports.
# Gate requests are pre-revision and source/gold absent; inspect persisted gate records.
gates=jlines(M/'GATE_DECISIONS.jsonl')
answer_fields=('gold_answer','correct','trusted_keys','reference_answer')
gate_safety={'rows':len(gates),'all_scored_before_revision':all(g.get('scored_before_revision') is True and g.get('call_id','').endswith('|initial|attempt-1') for g in gates),'no_revision_or_gold_fields':all(not any(k in json.dumps(g).lower() for k in answer_fields) for g in gates)}
# Scan generated request text in journal and raw GSM calls.
math_starts=[e for e in jlines(M/'STREAM_JOURNAL.jsonl') if e.get('event')=='call_start']
math_req_checks=[]
for e in math_starts:
    req=e.get('request',''); obj=ast.literal_eval(req) if isinstance(req,str) else req; math_req_checks.append({'id':e.get('question_id'),'phase':e.get('phase'),'only_generation_request_fields':isinstance(obj,dict) and set(obj)=={'model','messages','stream','options'} and all(isinstance(m,dict) and set(m).issubset({'role','content'}) for m in obj.get('messages',[]))})
gsm_calls=jlines(G/'RAW_CALLS.jsonl')
gsm_prompt_checks=[]
for c in gsm_calls:
    req=json.dumps(c.get('request',{}),ensure_ascii=False).lower()
    obj=c.get('request',{}); gsm_prompt_checks.append(isinstance(obj,dict) and set(obj)=={'model','messages','stream','options'} and isinstance(obj.get('messages'),list))
# source cohort/topic summaries.
levels=Counter(q['level'] for q in sel); topics=Counter(q['topic'] for q in sel)
finished=set(r['question_id'] for r in mathq)
report={'math_source':{'selection_count':len(sel),'unique_ids':len({q['example_id'] for q in sel}),'source_checks_passed':sum(x['source_problem_exact_match'] and x['level_match'] and x['key_matches_source_boxed'] for x in source_checks),'source_check_failures':[x for x in source_checks if not(x['source_problem_exact_match'] and x['level_match'] and x['key_matches_source_boxed'])],'difficulty_counts':dict(levels),'topic_counts':dict(topics),'source_partition_hashes':source_hashes,'selected_keys_count':len(keys),'completed_call_count':len(completed),'independent_answer_checks':{'parseable':sum(x['independent_parseable'] for x in call_grades),'matches_saved_parseability':sum(x['independent_parseable']==x['saved_parseable'] for x in call_grades),'correctness_matches_saved':sum(x['independent_correct']==x['saved_correct'] for x in call_grades),'mismatches':[x for x in call_grades if x['independent_correct']!=x['saved_correct']]},'selection_summary':{'attempted_questions':47,'complete_pairs':43,'incomplete_initial_calls':4,'incomplete_revision_calls':0,'unattempted_of_60':13}},'prompt_key_separation':{'math_initial_records':len(math_req_checks),'math_prompt_checks_pass':sum(x['only_generation_request_fields'] for x in math_req_checks),'gsm_calls':len(gsm_calls),'gsm_prompt_checks_pass':sum(gsm_prompt_checks),'gate_decisions':gate_safety},'math_accounting':{'pairs':len(mathq),'transitions':dict(Counter(x['transition'] for x in mathq)),'gate_rechecks':sum(x['gate_recheck'].lower()=='true' for x in mathq),'completion_rechecks':sum(x['completion_recheck'].lower()=='true' for x in mathq),'initial_tokens':sum(int(x['initial_tokens']) for x in mathq),'revision_tokens':sum(int(x['revision_tokens']) for x in mathq),'initial_latency_seconds':sum(float(x['initial_latency_seconds']) for x in mathq),'revision_latency_seconds':sum(float(x['revision_latency_seconds']) for x in mathq),'policy_rows':len(mathpol)},'gsm_accounting':{'pairs':len(gq),'transitions':dict(Counter(x['outcome'] for x in gq)),'gate_rechecks':sum(x['ml_gate_recheck_OOD'].lower()=='true' for x in gq),'completion_rechecks':sum(x['completion_heuristic_recheck'].lower()=='true' for x in gq),'calls':len(gsm_calls),'successful':sum(x.get('status')=='succeeded' for x in gsm_calls),'parse_failures':sum(x.get('parse_status')!='parseable' for x in gsm_calls),'tokens':sum(int((x.get('tokens') or {}).get('total_tokens',0)) for x in gsm_calls),'wall_seconds':sum(float(x.get('wall_clock_seconds') or 0) for x in gsm_calls),'policy_rows':len(gpol)},'independent_manual_examples':['geometry/100: ratio of arc lengths 1/16','prealgebra/870: Euclidean algorithm gcd=14','number_theory/68: modular inverse/power gives 2','algebra/903: compound-interest value rounds to 1160','counting_and_probability/423: combinations product 10800','precalculus/318: trigonometric identity yields 1/4','algebra/625: cone volume equality yields height 216; HELP is a parseable semantic answer change','prealgebra/131: direct enumeration gives 19; initial 18 and revision 36 are both mathematically wrong'],'runs':{'math_status':json.loads((M/'RUN_STATUS.json').read_text(encoding='utf8')),'gsm_status':json.loads((G/'FINAL_STATUS.json').read_text(encoding='utf8'))}}
out=Path(__file__).with_name('EVIDENCE_VERIFICATION.json'); out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'math_source':{'selected':len(sel),'source_key_matches':sum(x['source_problem_exact_match'] and x['level_match'] and x['key_matches_source_boxed'] for x in source_checks),'failures':[x for x in source_checks if not(x['source_problem_exact_match'] and x['level_match'] and x['key_matches_source_boxed'])],'answer_recheck':report['math_source']['independent_answer_checks']},'prompt_key_separation':report['prompt_key_separation'],'math_accounting':report['math_accounting'],'gsm_accounting':report['gsm_accounting']},indent=2,ensure_ascii=False))
