"""Meaningful protocol checks. Test fixtures are not research results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from rechecklab.collector import collect_pairs, parse_answer
from rechecklab.data import assert_disjoint, group_partition, transition_label, validate_pairs
from rechecklab.experiment import run_experiment
from rechecklab.features import extract_features
from rechecklab.metrics import policy_metrics


def pairs(prefix="question",count=60):
    records=[]
    for i in range(count):
        state=i%3
        records.append(dict(example_id=str(i),dataset=prefix,query=f"{prefix} problem {i}",
            base_response=f"Attempt {i}. {'Maybe' if state==0 else 'Check'} the calculation.",
            base_answer=str(i),base_correct=state!=0,final_correct=state!=2,
            base_actual_tokens=50+i,action_actual_tokens=30,base_done_reason="length" if state==0 else "stop",
            base_finalizer_used=False,action="recheck",model="test-only",recheck_protocol="test-only"))
    return validate_pairs(pd.DataFrame(records))


class ResearchSafeguards(unittest.TestCase):
    def test_outcomes_use_correctness_not_wording(self):
        self.assertEqual(transition_label(False,True),"HELP")
        self.assertEqual(transition_label(True,False),"HURT")
        self.assertEqual(transition_label(False,False),"NO_CHANGE")
        self.assertEqual(transition_label(True,True),"NO_CHANGE")
        with self.assertRaises(ValueError):
            transition_label("unknown",True)

    def test_future_outputs_and_labels_cannot_change_features(self):
        original=pairs(count=6)
        mutated=original.copy()
        mutated["base_correct"]=False
        mutated["final_correct"]=False
        mutated["label"]="HURT"
        mutated["intervention_response"]="GOLD ANSWER "*500
        mutated["final_answer"]="999999"
        mutated["action_actual_tokens"]=999999
        mutated["helpful_fix"]=True
        pd.testing.assert_frame_equal(extract_features(original),extract_features(mutated))

    def test_duplicate_question_and_rollouts_stay_together(self):
        frame=pairs(count=60)
        repeated=frame.iloc[:12].copy()
        repeated["rollout"]=1
        frame=pd.concat([frame,repeated],ignore_index=True)
        partitions=group_partition(frame,(.6,.2,.2),42)
        assert_disjoint(*partitions)
        self.assertEqual(sum(len(p) for p in partitions),len(frame))
        for group in repeated.group_id:
            self.assertEqual(sum(group in set(part.group_id) for part in partitions),1)

    def test_question_overlap_is_rejected(self):
        frame=pairs(count=20)
        with self.assertRaisesRegex(ValueError,"leakage"):
            assert_disjoint(frame.iloc[:10],frame.iloc[5:15])

    def test_inconsistent_initial_rollouts_are_rejected(self):
        frame=pairs(count=6)
        repeated=frame.iloc[:1].copy()
        repeated["base_correct"]=not bool(repeated.iloc[0].base_correct)
        with self.assertRaisesRegex(ValueError,"inconsistent"):
            validate_pairs(pd.concat([frame,repeated]))

    def test_metrics_weight_questions_not_number_of_rollouts(self):
        frame=validate_pairs(pd.DataFrame([
            dict(query="q1",example_id="1",base_response="a",base_correct=True,final_correct=True,base_actual_tokens=10,action_actual_tokens=2,
                 base_answer="a",base_done_reason="stop",base_finalizer_used=False,action="recheck",model="test-only",recheck_protocol="test-only"),
            dict(query="q1",example_id="1",base_response="a",base_correct=True,final_correct=False,base_actual_tokens=10,action_actual_tokens=4,
                 base_answer="a",base_done_reason="stop",base_finalizer_used=False,action="recheck",model="test-only",recheck_protocol="test-only"),
            dict(query="q2",example_id="2",base_response="b",base_correct=False,final_correct=True,base_actual_tokens=20,action_actual_tokens=6,
                 base_answer="b",base_done_reason="stop",base_finalizer_used=False,action="recheck",model="test-only",recheck_protocol="test-only"),
        ]))
        keep=policy_metrics(frame,[False,False,False])
        revise=policy_metrics(frame,[True,True,True])
        self.assertAlmostEqual(keep["accuracy_pct"],50)
        self.assertAlmostEqual(keep["average_total_tokens"],15)
        self.assertAlmostEqual(revise["accuracy_pct"],75)
        self.assertAlmostEqual(revise["harmful_flip_pct_of_all_questions"],25)
        self.assertAlmostEqual(revise["average_total_tokens"],19.5)

    def test_final_answer_parser_does_not_score_intermediate_numbers(self):
        self.assertIsNone(parse_answer("First I multiply 7 by 12, obtaining 84, then reconsider.","number"))
        self.assertEqual(parse_answer("Work\nFINAL: $1,000.00","number"),"1E+3")
        self.assertEqual(parse_answer("FINAL: -3","number"),"-3")
        self.assertEqual(parse_answer("FINAL: yes","boolean"),"yes")
        self.assertEqual(parse_answer("FINAL: B","choice"),"B")
        self.assertIsNone(parse_answer("I am unsure whether yes or no is appropriate.","boolean"))

    def test_target_test_labels_do_not_affect_fitting_or_adaptation(self):
        source,target=pairs("source"),pairs("target")
        _,holdout=group_partition(target,(.4,.6),43)
        changed=target.copy()
        mask=changed.group_id.isin(holdout.group_id)
        changed.loc[mask,"final_correct"]=~changed.loc[mask,"final_correct"]
        changed=validate_pairs(changed)
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/"a",Path(directory)/"b"
            meta_a,_,_=run_experiment(a,source,target,sizes=(10,20),repeats=1)
            meta_b,_,_=run_experiment(b,source,changed,sizes=(10,20),repeats=1)
            self.assertEqual(meta_a["model"],meta_b["model"])
            self.assertEqual(meta_a["source_threshold"],meta_b["source_threshold"])
            pred_a=pd.read_csv(a/"reports"/"target_predictions.csv")
            pred_b=pd.read_csv(b/"reports"/"target_predictions.csv")
            for label in ("HELP","NO_CHANGE","HURT"):
                np.testing.assert_allclose(pred_a["p_"+label],pred_b["p_"+label])
            cal_a=pd.read_csv(a/"reports"/"adaptation_runs.csv")
            cal_b=pd.read_csv(b/"reports"/"adaptation_runs.csv")
            np.testing.assert_allclose(cal_a.threshold,cal_b.threshold)

    def test_collector_keeps_first_response_and_can_resume(self):
        class Tags:
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def read(self): return json.dumps({"models":[{"name":"qwen3:1.7b","digest":"mock-test-only"}]}).encode()
        answers=[dict(message={"content":"FINAL: 4"},done=True,prompt_eval_count=10,eval_count=5,done_reason="stop",wall_seconds=.01),
                 dict(message={"content":"Checked. FINAL: 4"},done=True,prompt_eval_count=20,eval_count=7,done_reason="stop",wall_seconds=.02)]
        with tempfile.TemporaryDirectory() as directory:
            inputs,output=Path(directory)/"questions.jsonl",Path(directory)/"pairs.jsonl"
            inputs.write_text(json.dumps(dict(example_id="x",dataset="unit-test-only",query="What is two plus two?",gold_answer="4",answer_type="number"))+"\n")
            with patch("rechecklab.collector.urlopen",return_value=Tags()),patch("rechecklab.collector.ollama_chat",side_effect=answers) as chat:
                self.assertEqual(collect_pairs(inputs,output),1)
                second_messages=chat.call_args_list[1].args[1]
                self.assertEqual(second_messages[1]["content"],"FINAL: 4")
                self.assertNotIn("gold_answer",json.dumps(second_messages))
            with patch("rechecklab.collector.urlopen",return_value=Tags()),patch("rechecklab.collector.ollama_chat") as chat:
                self.assertEqual(collect_pairs(inputs,output),0)
                chat.assert_not_called()
            row=json.loads(output.read_text())
            self.assertEqual(row["label"],"NO_CHANGE")
            self.assertEqual(row["action_actual_tokens"],27)
            with self.assertRaisesRegex(ValueError,"settings differ"):
                collect_pairs(inputs,output,budget=512)


if __name__=="__main__":
    unittest.main()
