"""Read-only integration checks for the frozen recovery workspace."""
import hashlib,json,pathlib,sys,unittest
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
EVAL=ROOT/"outputs/live_study/v0.4.3_math_l4_l5_frozen_evaluation_2026-11-16_01"
OLD=ROOT/"outputs/live_study/v0.4.2_math_l4_l5_frozen_evaluation_2026-11-16_01"
sys.path.insert(0,str(HERE));from durable_stream import JsonlJournal,reconstruct_calls,next_attempt
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

class RecoveryWorkspaceTests(unittest.TestCase):
    def test_selection_and_order_are_verbatim_frozen_copy(self):
        self.assertEqual(sha(EVAL/"QUESTION_SELECTION.jsonl"),sha(OLD/"QUESTION_SELECTION.jsonl"))
        self.assertEqual(sha(EVAL/"TRUSTED_KEYS.jsonl"),sha(OLD/"TRUSTED_KEYS.jsonl"))
        self.assertEqual(len(rows(EVAL/"QUESTION_SELECTION.jsonl")),60)
    def test_legacy_first_attempt_remains_distinct_and_incomplete(self):
        calls=reconstruct_calls(EVAL/"STREAM_JOURNAL.jsonl")
        first=[c for c in calls if c["question_id"]=="math:test:intermediate_algebra:intermediate_algebra/823" and c["attempt_number"]==1]
        self.assertEqual(len(first),1);self.assertEqual(first[0]["status"],"incomplete")
        self.assertEqual(first[0]["content"],"");self.assertIsNone(first[0]["completion_reason"]);self.assertIsNone(first[0]["token_statistics"])
        self.assertEqual(next_attempt(JsonlJournal.read_events(EVAL/"STREAM_JOURNAL.jsonl"),first[0]["question_id"],"initial"),2)
    def test_59_other_questions_remain_unattempted(self):
        qids={x["example_id"] for x in rows(EVAL/"QUESTION_SELECTION.jsonl")}
        events=JsonlJournal.read_events(EVAL/"STREAM_JOURNAL.jsonl")
        attempted={x["question_id"] for x in events if x.get("event")=="call_start"}
        self.assertEqual(len(qids-attempted),59)
    def test_protocol_and_gate_are_frozen(self):
        manifest=json.loads((EVAL/"EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["protocol"]["num_ctx"],8192)
        self.assertEqual(manifest["protocol"]["num_predict"],-1)
        self.assertEqual(manifest["protocol"]["model"]["digest"],"0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0")
        old=json.loads((OLD/"EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))["protocol"]
        self.assertEqual(manifest["protocol"]["prompts"],old["prompts"])
        self.assertEqual(manifest["protocol"]["gate_threshold"],old["gate_threshold"])
        self.assertEqual(manifest["protocol"]["gate_artifact_sha256"],old["gate_artifact_sha256"])
        self.assertEqual(manifest["protocol"]["feature_schema"],old["feature_schema"])
        self.assertEqual(manifest["protocol"]["generation"]["num_ctx"],8192)
        self.assertEqual(manifest["protocol"]["generation"]["num_predict"],-1)
        self.assertEqual(sha(ROOT/"outputs/v0.2/models/source_gate.joblib"),"9a92e4e9a91d1a05e0ead0dd638f434838e9b944e406ff01fa3d82ae91e3c439")

if __name__=="__main__":unittest.main(verbosity=2)
