"""Question-clustered, read-only analysis of frozen RecheckLab v0.2 records.

No model fitting, threshold selection, label edits, or inference occur here.
All reported policies use frozen artifact probabilities and recorded outcomes.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, log_loss

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rechecklab.data import LABELS, question_hash, validate_pairs
from rechecklab.predictor import load_artifact, decide_recheck
from rechecklab.metrics import policy_metrics, predictor_metrics

SEED = 20261116
BOOT = 10000
ARTIFACT = ROOT / "outputs/v0.2/models/source_gate.joblib"
PREP = ROOT / "outputs/v0.2/data/prepared"
RAW = ROOT / "data/raw"
SAVED_PREDICTIONS = ROOT / "outputs/ml_efficiency_evaluation/evaluation_2026-11-03/analysis_predictions.csv"

def write_csv(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)

def qgroups(df):
    return df.groupby("group_id", sort=True).indices

def cluster_means(values, groups):
    tmp = pd.DataFrame({"g": groups, "v": values})
    return tmp.groupby("g", sort=True).v.mean().to_numpy(float)

def boot_ci(values, groups, seed=SEED, repeats=BOOT):
    x = cluster_means(np.asarray(values, float), np.asarray(groups))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(repeats, len(x)))
    m = x[draws].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))

def paired_test(values, groups, seed=SEED, repeats=20000):
    # Two-sided cluster sign-flip randomization test for the paired mean difference.
    x = cluster_means(np.asarray(values, float), np.asarray(groups))
    obs = abs(x.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, repeats, 1000):
        n = min(1000, repeats-start)
        signs = rng.choice(np.array([-1., 1.]), size=(n, len(x)))
        exceed += int((np.abs((signs*x).mean(axis=1)) >= obs-1e-15).sum())
    return (exceed+1)/(repeats+1)

def load_prepared(name):
    return validate_pairs(pd.read_parquet(PREP / f"{name}.parquet"))

def load_external(name, artifact_protocol):
    raw = pd.read_parquet(RAW / f"{name}.parquet")
    f = raw[raw.action.eq("active_verify")].copy().reset_index(drop=True)
    f["dataset"] = "SEVRA_" + name
    f["recheck_protocol"] = artifact_protocol["recheck_protocol"]
    f.loc[f.base_done_reason.eq(""), "base_done_reason"] = "unknown"
    f = validate_pairs(f)
    return f

def policy_map(df, artifact, cohort):
    recomputed = artifact["predictor"].probabilities(df)
    partition={"MATH source development":"source_development","MATH source test":"source_test",
               "CommonsenseQA target test":"target_test"}.get(cohort)
    if partition:
        saved=pd.read_csv(SAVED_PREDICTIONS)
        saved=saved[saved.partition.eq(partition)].sort_values("row_index")
        if len(saved)!=len(df) or not np.array_equal(saved.group_id.to_numpy(),df.group_id.to_numpy()) or not np.array_equal(saved.label.to_numpy(),df.label.to_numpy()):
            raise ValueError(f"Saved predictions do not align exactly with {cohort} records.")
        proba=saved[["p_HELP","p_NO_CHANGE","p_HURT"]].to_numpy(float)
        if not np.allclose(proba,recomputed,rtol=0,atol=1e-12):
            raise ValueError(f"Saved predictions do not reproduce under the frozen artifact for {cohort}.")
        saved_gate=saved["Frozen ML gate"].astype(str).str.lower().map({"true":True,"false":False}).to_numpy(bool)
        if not np.array_equal(saved_gate,decide_recheck(proba,artifact["threshold"],artifact["harm_weight"])):
            raise ValueError(f"Saved policy decisions do not reproduce for {cohort}.")
    else:
        proba=recomputed
    pred = np.array(LABELS)[proba.argmax(axis=1)]
    gate = decide_recheck(proba, artifact["threshold"], artifact["harm_weight"])
    completion = df.base_done_reason.eq("length").to_numpy(bool)
    return proba, pred, {"Never": np.zeros(len(df), bool), "Always": np.ones(len(df), bool),
                         "Completion": completion, "Frozen ML": np.asarray(gate, bool)}

def policy_rows(df, policies, cohort):
    rows=[]
    for policy, decisions in policies.items():
        m=policy_metrics(df, decisions)
        groups=df.group_id.to_numpy(); n=df.group_id.nunique()
        helpv=(decisions & df.label.eq("HELP").to_numpy()).astype(float)
        hurtv=(decisions & df.label.eq("HURT").to_numpy()).astype(float)
        hci=boot_ci(helpv,groups); uci=boot_ci(hurtv,groups,SEED+17)
        acc=np.where(decisions,df.final_correct,df.base_correct).astype(float)
        aci=boot_ci(acc,groups,SEED+23)
        base_mean=float(cluster_means(df.base_actual_tokens.to_numpy(float),groups).mean())
        always_mean=float(cluster_means(df.base_actual_tokens.to_numpy(float)+df.action_actual_tokens.to_numpy(float),groups).mean())
        total_help=float(cluster_means(df.label.eq("HELP").to_numpy(float),groups).sum())
        total_hurt=float(cluster_means(df.label.eq("HURT").to_numpy(float),groups).sum())
        selected_help=m["helpful_fix_pct_of_all_questions"]*n/100
        selected_hurt=m["harmful_flip_pct_of_all_questions"]*n/100
        selected_calls=m["recheck_pct"]*n/100
        rows.append({"cohort":cohort,"evaluation_status": status(cohort),"policy":policy,**m,
          "average_initial_tokens":base_mean,
          "average_additional_recheck_tokens":m["average_extra_tokens"],
          "tokens_saved_vs_always_pct":100*(1-m["average_total_tokens"]/always_mean) if always_mean else np.nan,
          "helpful_fix_question_equivalents":selected_help,"harmful_flip_question_equivalents":selected_hurt,
          "helpful_fixes_missed_question_equivalents":max(0,total_help-selected_help),
          "harmful_revisions_avoided_question_equivalents":max(0,total_hurt-selected_hurt),
          "all_help_question_equivalents":total_help,"all_hurt_question_equivalents":total_hurt,
          "help_capture_rate_pct":100*selected_help/total_help if total_help else np.nan,
          "helpful_recheck_precision_pct":100*selected_help/selected_calls if selected_calls else np.nan,
          "recheck_calls_avoided_question_equivalents":n-selected_calls,
          "accuracy_change_vs_never_pp":m["accuracy_pct"]-float(cluster_means(df.base_correct.to_numpy(float),groups).mean()*100),
          "helpful_rate_cluster_ci_low_pct":100*hci[0],"helpful_rate_cluster_ci_high_pct":100*hci[1],
          "harmful_rate_cluster_ci_low_pct":100*uci[0],"harmful_rate_cluster_ci_high_pct":100*uci[1],
          "accuracy_cluster_ci_low_pct":100*aci[0],"accuracy_cluster_ci_high_pct":100*aci[1]})
    # Hindsight oracle is one rollout-level best choice, explicitly nondeployable.
    oracle=df.label.eq("HELP").to_numpy(bool)
    m=policy_metrics(df, oracle)
    n=df.group_id.nunique()
    groups=df.group_id.to_numpy(); base_mean=float(cluster_means(df.base_actual_tokens.to_numpy(float),groups).mean())
    always_mean=float(cluster_means(df.base_actual_tokens.to_numpy(float)+df.action_actual_tokens.to_numpy(float),groups).mean())
    total_help=float(cluster_means(df.label.eq("HELP").to_numpy(float),groups).sum())
    total_hurt=float(cluster_means(df.label.eq("HURT").to_numpy(float),groups).sum())
    selected_calls=m["recheck_pct"]*n/100
    rows.append({"cohort":cohort,"evaluation_status":status(cohort),"policy":"Oracle (hindsight)",**m,
      "average_initial_tokens":base_mean,"average_additional_recheck_tokens":m["average_extra_tokens"],
      "tokens_saved_vs_always_pct":100*(1-m["average_total_tokens"]/always_mean) if always_mean else np.nan,
      "helpful_fix_question_equivalents":m["helpful_fix_pct_of_all_questions"]*n/100,
      "harmful_flip_question_equivalents":0.0,"helpful_fixes_missed_question_equivalents":0.0,
      "harmful_revisions_avoided_question_equivalents":total_hurt,"all_help_question_equivalents":total_help,
      "all_hurt_question_equivalents":total_hurt,"help_capture_rate_pct":100.0 if total_help else np.nan,
      "helpful_recheck_precision_pct":100*total_help/selected_calls if selected_calls else np.nan,
      "recheck_calls_avoided_question_equivalents":n-selected_calls,
      "accuracy_change_vs_never_pp":m["accuracy_pct"]-float(cluster_means(df.base_correct.to_numpy(float),groups).mean()*100),
      "helpful_rate_cluster_ci_low_pct":np.nan,
      "helpful_rate_cluster_ci_high_pct":np.nan,"harmful_rate_cluster_ci_low_pct":np.nan,
      "harmful_rate_cluster_ci_high_pct":np.nan,"accuracy_cluster_ci_low_pct":np.nan,"accuracy_cluster_ci_high_pct":np.nan})
    return rows

def status(cohort):
    return {"MATH source test":"historical test; repeatedly inspected/exploratory",
            "CommonsenseQA target test":"historical test; repeatedly inspected/exploratory",
            "MATH500":"published external split; exploratory diagnostic, not untouched confirmation",
            "GSM8K":"published external split; exploratory diagnostic, not untouched confirmation",
            "MATH source development":"previously inspected development; descriptive confirmation"}.get(cohort,"historical diagnostic")

def classifier_rows(df, proba, cohort):
    m=predictor_metrics(df,proba,bootstrap_repeats=10000,seed=SEED)
    truth=df.label.to_numpy(); pred=np.array(LABELS)[proba.argmax(axis=1)]
    w=1/df.groupby("group_id").group_id.transform("size").to_numpy()
    precision,recall,f1,support=precision_recall_fscore_support(truth,pred,labels=LABELS,sample_weight=w,zero_division=0)
    weighted_f1=float(np.average(f1,weights=support))
    cm_raw=confusion_matrix(truth,pred,labels=LABELS)
    cm_question=confusion_matrix(truth,pred,labels=LABELS,sample_weight=w)
    rows=[]
    for i,c in enumerate(LABELS):
        rows.append({"cohort":cohort,"evaluation_status":status(cohort),"metric_type":"class","class":c,
            "precision":precision[i],"recall":recall[i],"f1":f1[i],"raw_support":int((truth==c).sum()),
            "question_equivalent_support":float(support[i]),"accuracy":np.nan,"macro_f1":np.nan,
            "weighted_f1":np.nan,"log_loss":np.nan,"ci_low":np.nan,"ci_high":np.nan})
    rows += [{"cohort":cohort,"evaluation_status":status(cohort),"metric_type":"overall","class":"accuracy",
      "precision":np.nan,"recall":np.nan,"f1":np.nan,"raw_support":len(df),"question_equivalent_support":df.group_id.nunique(),
      "accuracy":m["accuracy"],"macro_f1":m["macro_f1"],"weighted_f1":weighted_f1,"log_loss":m["log_loss"],"ci_low":np.nan,"ci_high":np.nan,
      "macro_f1_ci_low":m["macro_f1_95pct_ci"][0],"macro_f1_ci_high":m["macro_f1_95pct_ci"][1],
      "log_loss_ci_low":m["log_loss_95pct_ci"][0],"log_loss_ci_high":m["log_loss_95pct_ci"][1]}]
    for i,a in enumerate(LABELS):
        for j,b in enumerate(LABELS):
            rows.append({"cohort":cohort,"evaluation_status":status(cohort),"metric_type":"confusion","class":f"true_{a}_pred_{b}",
              "precision":np.nan,"recall":np.nan,"f1":np.nan,"raw_support":int(cm_raw[i,j]),"question_equivalent_support":float(cm_question[i,j]),
              "accuracy":np.nan,"macro_f1":np.nan,"weighted_f1":np.nan,"log_loss":np.nan,"ci_low":np.nan,"ci_high":np.nan})
    return rows

def main():
    global OUT
    parser=argparse.ArgumentParser(description="Reproduce frozen v0.2 policy metrics into a new, empty run directory.")
    parser.add_argument("--output-dir",type=Path,default=OUT,
      help="Fresh empty output directory; defaults to this run, which will refuse overwrite after its first execution.")
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty analysis run directory: {OUT}; choose a new versioned --output-dir.")
    OUT.mkdir(parents=True,exist_ok=True)
    artifact=load_artifact(ARTIFACT)
    protocol=artifact["protocol"]
    cohorts={"MATH source development":load_prepared("source_development"),
             "MATH source test":load_prepared("source_test"),
             "CommonsenseQA target test":load_prepared("target_test")}
    external={n:load_external(n,protocol) for n in ("math500","gsm8k")}
    cohorts["MATH500"]=external["math500"]; cohorts["GSM8K"]=external["gsm8k"]
    # Duplicate-question and cross-cohort overlap diagnostics; never de-duplicate outcomes silently.
    overlap=[]
    for name,df in cohorts.items():
        overlap.append({"cohort":name,"questions":df.group_id.nunique(),"rows":len(df),
          "duplicate_rollout_groups":int((df.groupby("group_id").size()>1).sum()),
          "max_rollouts_per_question":int(df.groupby("group_id").size().max()),
          "missing_completion_rows":int(df.base_done_reason.eq("unknown").sum())})
    names=list(cohorts)
    for i,a in enumerate(names):
      for b in names[i+1:]:
        overlap.append({"cohort":f"OVERLAP:{a} vs {b}","questions":len(set(cohorts[a].group_id)&set(cohorts[b].group_id)),"rows":np.nan,
          "duplicate_rollout_groups":np.nan,"max_rollouts_per_question":np.nan,"missing_completion_rows":np.nan})
    for part in ("source_train","target_adaptation_pool"):
      excluded=load_prepared(part)
      for name,df in cohorts.items():
        overlap.append({"cohort":f"OVERLAP:{part} vs {name}","questions":len(set(excluded.group_id)&set(df.group_id)),"rows":np.nan,
          "duplicate_rollout_groups":np.nan,"max_rollouts_per_question":np.nan,"missing_completion_rows":np.nan})
    # Explicit saved-action identity checks against the frozen protocol contract.
    expected_action_hash=protocol.get("action_description_sha256")
    for name,df in cohorts.items():
        descriptions=df.action_description.astype(str).drop_duplicates().tolist() if "action_description" in df else []
        actual_hashes=[hashlib.sha256(v.encode("utf-8")).hexdigest() for v in descriptions]
        overlap.append({"cohort":f"PROTOCOL:{name}","questions":np.nan,"rows":np.nan,
          "duplicate_rollout_groups":np.nan,"max_rollouts_per_question":np.nan,"missing_completion_rows":np.nan,
          "action_description_hash_matches_frozen":bool(actual_hashes and actual_hashes==[expected_action_hash]),
          "model_matches_frozen":bool("model" in df and df.model.eq(protocol["model"]).all()),
          "action_matches_frozen":bool("action" in df and df.action.eq(protocol["action"]).all())})
    write_csv(overlap,"DATASET_PROTOCOL_COMPARISON.csv")
    allpol=[]; allcls=[]; confusion=[]; pairrows=[]; completions=[]; agreements=[]
    percohort={}
    for ix,(name,df) in enumerate(cohorts.items()):
        proba,pred,policies=policy_map(df,artifact,name)
        percohort[name]=(df,policies,proba,pred)
        allpol+=policy_rows(df,policies,name)
        allcls+=classifier_rows(df,proba,name)
        disagree=policies["Frozen ML"]!=policies["Completion"]
        for cls in LABELS:
          mask=disagree & df.label.eq(cls).to_numpy()
          agreements.append({"cohort":name,"evaluation_status":status(name),"decision":"Frozen ML vs completion",
            "different_rollout_rows":int(disagree.sum()),"different_question_clusters":int(df.loc[disagree,"group_id"].nunique()),
            "disagreement_label":cls,"rows_in_disagreement_class":int(mask.sum()),
            "question_clusters_in_disagreement_class":int(df.loc[mask,"group_id"].nunique()),
            "decision_agreement_pct_rows":100*(1-disagree.mean())})
        weights=1/df.groupby("group_id").group_id.transform("size").to_numpy()
        cmraw=confusion_matrix(df.label.to_numpy(),pred,labels=LABELS)
        cmq=confusion_matrix(df.label.to_numpy(),pred,labels=LABELS,sample_weight=weights)
        for i,a in enumerate(LABELS):
          for j,b in enumerate(LABELS): confusion.append({"cohort":name,"true":a,"predicted":b,
            "raw_rollout_count":int(cmraw[i,j]),"question_equivalent_count":float(cmq[i,j])})
        # Question-cluster paired differences: means of rollout outcomes within each question.
        groups=df.group_id.to_numpy()
        def outcome(dec): return np.where(dec,df.final_correct.to_numpy(float),df.base_correct.to_numpy(float))
        for pol in ("Always","Completion","Frozen ML"):
          for ref in ("Never","Completion"):
            if pol==ref: continue
            d=outcome(policies[pol])-outcome(policies[ref])
            lo,hi=boot_ci(d,groups,SEED+ix)
            pairrows.append({"cohort":name,"evaluation_status":status(name),"comparison":f"{pol} - {ref}",
              "paired_accuracy_difference_pp":100*float(cluster_means(d,groups).mean()),"cluster_bootstrap_95ci_low_pp":100*lo,
              "cluster_bootstrap_95ci_high_pp":100*hi,"cluster_signflip_p_unadjusted":paired_test(d,groups,SEED+ix),
              "independent_question_clusters":df.group_id.nunique(),"rollout_rows":len(df),
              "interpretation":"Descriptive paired randomization; no multiplicity adjustment; exposed cohorts exploratory."})
        # Cost and paired cost difference all selected against always policy.
        for pol,dec in policies.items():
          if pol=="Always": continue
          d=(dec.astype(float)-np.ones(len(df)))*df.action_actual_tokens.to_numpy(float)
          lo,hi=boot_ci(d,groups,SEED+ix+100)
          pairrows.append({"cohort":name,"evaluation_status":status(name),"comparison":f"{pol} - Always additional tokens",
            "paired_accuracy_difference_pp":np.nan,"cluster_bootstrap_95ci_low_pp":np.nan,"cluster_bootstrap_95ci_high_pp":np.nan,
            "cluster_signflip_p_unadjusted":np.nan,"independent_question_clusters":df.group_id.nunique(),"rollout_rows":len(df),
            "mean_token_difference":float(cluster_means(d,groups).mean()),"cluster_bootstrap_95ci_low_tokens":lo,
            "cluster_bootstrap_95ci_high_tokens":hi,"interpretation":"Paired question-cluster token difference versus always rechecking."})
        # Completion-state class/outcome distributions: non-completion group is length; completed=stop.
        for state,mask in [("length",df.base_done_reason.eq("length")),("stop",df.base_done_reason.eq("stop")),("unknown",df.base_done_reason.eq("unknown"))]:
          sub=df.loc[mask]
          if sub.empty: continue
          counts=sub.label.value_counts().to_dict()
          completions.append({"cohort":name,"completion_state":state,"rows":len(sub),"questions":sub.group_id.nunique(),
            "HELP_rows":counts.get("HELP",0),"NO_CHANGE_rows":counts.get("NO_CHANGE",0),"HURT_rows":counts.get("HURT",0),
            "HELP_rate_rows_pct":100*counts.get("HELP",0)/len(sub),"HURT_rate_rows_pct":100*counts.get("HURT",0)/len(sub),
            "questions_equal_help_rate_pct":100*float(cluster_means(sub.label.eq("HELP").astype(float),sub.group_id).mean()),
            "questions_equal_hurt_rate_pct":100*float(cluster_means(sub.label.eq("HURT").astype(float),sub.group_id).mean())})
    # Flag observed Pareto-efficient points within each cohort (oracle remains hindsight-only).
    for row in allpol:
      if row["policy"] == "Oracle (hindsight)":
        row["attainable_pareto_frontier"] = "hindsight-only"
        continue
      peers=[x for x in allpol if x["cohort"]==row["cohort"] and x is not row and x["policy"]!="Oracle (hindsight)"]
      row["attainable_pareto_frontier"] = not any((x["accuracy_pct"] >= row["accuracy_pct"] and
          x["average_total_tokens"] <= row["average_total_tokens"] and
          (x["accuracy_pct"] > row["accuracy_pct"] or x["average_total_tokens"] < row["average_total_tokens"])) for x in peers)
    write_csv(allpol,"MAIN_POLICY_COMPARISON.csv")
    write_csv(allcls,"CLASSIFIER_METRICS_CONFUSION.csv")
    write_csv(confusion,"CONFUSION_MATRICES.csv")
    write_csv(pairrows,"PAIRED_COMPARISONS.csv")
    write_csv(completions,"HELP_HURT_COMPLETION_BREAKDOWN.csv")
    write_csv(agreements,"GATE_HEURISTIC_AGREEMENT.csv")
    # Compact descriptive outcome-rate figure by recorded completion state.
    states=pd.DataFrame(completions)
    fig,ax=plt.subplots(figsize=(10.5,5.0))
    x=np.arange(len(states)); width=.36
    ax.bar(x-width/2,states.questions_equal_help_rate_pct,width,label="HELP")
    ax.bar(x+width/2,states.questions_equal_hurt_rate_pct,width,label="HURT")
    short={"MATH source development":"MATH dev","MATH source test":"MATH test",
           "CommonsenseQA target test":"CSQA target","MATH500":"MATH500","GSM8K":"GSM8K"}
    ax.set_xticks(x, [f"{short[r.cohort]}\n{r.completion_state}" for r in states.itertuples()], ha="center",fontsize=8)
    ax.set_ylabel("Question-equal event rate (%)"); ax.set_title("Recorded HELP/HURT outcomes by initial completion state")
    ax.grid(axis="y",alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(OUT/"help_hurt_by_completion.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    # Pareto figures: each policy point is a frozen observed policy; costs include neutral rollouts.
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharey=True)
    styles={"Never":("#1f77b4","o"),"Always":("#ff7f0e","s"),"Completion":("#2ca02c","^"),
            "Frozen ML":("#d62728","D"),"Oracle (hindsight)":("#9467bd","X")}
    for ax,(name,(df,policies,_,_)) in zip(axes.flat,percohort.items()):
      for row in (r for r in allpol if r["cohort"]==name):
        color,marker=styles[row["policy"]]
        ax.scatter(row["average_total_tokens"],row["accuracy_pct"],s=64,c=color,marker=marker,
                   edgecolors="black",linewidths=.35,alpha=.78,zorder=3)
      ax.set_title(name,fontsize=10); ax.set_xlabel("Recorded tokens / question"); ax.grid(alpha=.25)
    axes.flat[-1].set_visible(False)
    axes[0,0].set_ylabel("Final accuracy (%)"); axes[1,0].set_ylabel("Final accuracy (%)")
    handles=[plt.Line2D([0],[0],marker=m,color="w",label=k,markerfacecolor=c,markeredgecolor="black",markersize=8)
             for k,(c,m) in styles.items()]
    fig.suptitle("Frozen policy accuracy vs recorded token cost (question-equal)")
    fig.legend(handles=handles,loc="lower center",ncol=5,frameon=False)
    fig.tight_layout(rect=(0,.08,1,.95)); fig.savefig(OUT/"accuracy_vs_tokens.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    # Claims matrix linked directly to verified record/code evidence.
    claims=[
      ["Selective computation changes replay cost and sometimes accuracy", "MEASURED", "MAIN_POLICY_COMPARISON.csv", "Recorded paired responses only; no live latency"],
      ["ML beats completion", "NOT SUPPORTED", "MAIN_POLICY_COMPARISON.csv; PAIRED_COMPARISONS.csv", "CSQA tied; source MATH completion is at least as accurate; tests exposed"],
      ["MATH source is official MATH Level 5", "FALSE / provenance not established", "data/raw/DATASET_CARD.md; v0.2 split manifest", "It is a SEVRA MATH-training holdout"],
      ["MATH500 and GSM8K frozen gate metrics", "NEW EXTERNAL DIAGNOSTICS", "MAIN_POLICY_COMPARISON.csv", "Pinned SEVRA records; exploratory, protocol provenance omits exact prompts and checkpoint digest"],
      ["Token savings imply wall-time savings", "NOT MEASURED", "outputs/ml_efficiency_evaluation/evaluation_2026-11-03/timing_evidence.json", "No trustworthy per-call timing fields"],
      ["Classification is reliable for rare HURT", "NOT ESTABLISHED", "CLASSIFIER_METRICS_CONFUSION.csv", "Hurt supports are sparse; inspect recall/CI and abstain from safety claim"],
      ["Paired rollout outcomes are independent rows", "FALSE", "rechecklab/data.py:39,74; rechecklab/metrics.py:9,22", "Question clustered and equal weighted"],
    ]
    pd.DataFrame(claims,columns=["claim","status","evidence_file","scope_or_limit"]).to_csv(OUT/"CLAIMS_EVIDENCE.csv",index=False)
    # Hash frozen model and input files; values establish what was read by this run.
    files=[ARTIFACT,SAVED_PREDICTIONS,PREP/"source_development.parquet",PREP/"source_test.parquet",PREP/"target_test.parquet",
           RAW/"math500.parquet",RAW/"gsm8k.parquet",RAW/"DATASET_CARD.md",RAW/"release_manifest.json"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT/"INPUT_SHA256.json").write_text(json.dumps(hashes,indent=2)+"\n",encoding="utf-8")
    manifest={"run":"2026-11-16_01","analysis_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "artifact_sha256":hashes[str(ARTIFACT.relative_to(ROOT))],"artifact_protocol":protocol,
      "threshold":artifact["threshold"],"harm_weight":artifact["harm_weight"],"seed":SEED,"bootstrap_replicates":BOOT,
      "tests_used_for_selection":False,"model_training_run":False,"new_llm_inference_run":False,
      "existing_frozen_gate_scored_external_records":True,"primary_cohort_predictions":"reused from saved analysis_predictions.csv and checked against frozen artifact",
      "cohorts":{n:{"rows":len(d),"questions":d.group_id.nunique()} for n,d in cohorts.items()},
      "policy_definition":"completion = base_done_reason == length; ML uses frozen v0.2 probability and saved threshold/harm weight; oracle selects HELP rows hindsight only",
      "aggregation":"each question contributes weight one; rollout rows averaged within canonical question hash",
      "external_protocol_limit":"SEVRA card and pinned release establish model family/action and settings; exact prompts/checkpoint digest absent"}
    (OUT/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"outputs":str(OUT),"cohorts":manifest["cohorts"],"threshold":artifact["threshold"],"protocol":protocol},indent=2))

if __name__=="__main__": main()
