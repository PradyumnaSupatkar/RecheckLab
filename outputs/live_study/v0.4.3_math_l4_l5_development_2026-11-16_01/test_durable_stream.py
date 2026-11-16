import json, pathlib, tempfile, unittest
from durable_stream import DurableStreamCollector, JsonlJournal, reconstruct_calls, next_attempt

class DurableStreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=pathlib.Path(self.tmp.name)/"journal.jsonl"
        self.collector=DurableStreamCollector(self.path,heartbeat=lambda:{"cpu":0.2,"gpu_mib":100})
    def tearDown(self): self.tmp.cleanup()
    @staticmethod
    def chunk(content="",thinking="",**fields):
        return json.dumps({"message":{"role":"assistant","content":content,"thinking":thinking},**fields})+"\n"
    def test_natural_completion_and_exact_metadata(self):
        result=self.collector.consume([self.chunk("work ","private reasoning "),self.chunk("FINAL: 3",done=True,done_reason="stop",prompt_eval_count=4,eval_count=7)],
            question_id="q1",phase="initial",attempt_number=1,call_id="c1",request={"num_predict":-1})
        self.assertEqual(result["status"],"complete");self.assertEqual(result["content"],"work FINAL: 3")
        self.assertEqual(result["completion_reason"],"stop");self.assertEqual(result["token_statistics"]["eval_count"],7)
        self.assertEqual(result["thinking"],"private reasoning ")
        events=JsonlJournal.read_events(self.path);self.assertEqual([e["event"] for e in events], ["call_start","chunk","chunk","call_complete"])
        self.assertEqual([e["chunk_index"] for e in events if e["event"]=="chunk"],[1,2])
        self.assertTrue(all(e.get("timestamp_utc") for e in events));self.assertGreaterEqual(events[-2]["elapsed_monotonic_seconds"],0)
    def test_interruption_persists_and_restart_reconstructs_partial(self):
        def stream():
            yield self.chunk("partial ")
            raise KeyboardInterrupt("simulated disconnect")
        with self.assertRaises(KeyboardInterrupt): self.collector.consume(stream(),question_id="q1",phase="initial",attempt_number=1,call_id="c1")
        partial=reconstruct_calls(self.path)[0]
        self.assertEqual(partial["status"],"incomplete");self.assertEqual(partial["content"],"partial ")
        self.assertIsNone(partial["completion_reason"]);self.assertIsNone(partial["token_statistics"])
        self.assertEqual(next_attempt(JsonlJournal.read_events(self.path),"q1","initial"),2)
        restarted=DurableStreamCollector(self.path,heartbeat=lambda:{"restarted":True})
        restarted.consume([self.chunk("FINAL: 3",done=True,done_reason="stop",eval_count=2)],question_id="q1",phase="initial",attempt_number=2,call_id="c2")
        calls=reconstruct_calls(self.path);self.assertEqual([x["status"] for x in calls],["incomplete","complete"])
        self.assertEqual(calls[0]["content"],"partial ");self.assertEqual(calls[1]["content"],"FINAL: 3")
        self.assertEqual(next_attempt(JsonlJournal.read_events(self.path),"q1","initial"),3)
    def test_missing_metadata_is_missing_not_fabricated(self):
        result=self.collector.consume([self.chunk("FINAL: 3",done=True)],question_id="q",phase="revision",attempt_number=1,call_id="c")
        self.assertEqual(result["status"],"complete");self.assertIsNone(result["completion_reason"]);self.assertIsNone(result["token_statistics"])
    def test_disconnect_without_done_marks_incomplete(self):
        with self.assertRaises(ConnectionError): self.collector.consume([self.chunk("some text")],question_id="q",phase="initial",attempt_number=1,call_id="c")
        result=reconstruct_calls(self.path)[0]
        self.assertEqual(result["status"],"incomplete");self.assertEqual(result["content"],"some text")
        self.assertEqual(result["last_event"]["diagnostic_class"],"client_disconnected")
    def test_no_progress_timeout_is_distinguished(self):
        def stalled():
            yield self.chunk("first")
            raise TimeoutError("synthetic idle timeout")
        with self.assertRaises(TimeoutError):self.collector.consume(stalled(),question_id="q",phase="initial",attempt_number=1,call_id="c")
        result=reconstruct_calls(self.path)[0]
        self.assertEqual(result["status"],"incomplete");self.assertEqual(result["last_event"]["diagnostic_class"],"no_progress_stall")
    def test_malformed_json_chunk_is_preserved(self):
        with self.assertRaises(json.JSONDecodeError): self.collector.consume(["{not-json}\n"],question_id="q",phase="initial",attempt_number=1,call_id="c")
        events=JsonlJournal.read_events(self.path);bad=next(x for x in events if x["event"]=="malformed_stream_chunk")
        self.assertEqual(bad["raw_chunk_line"],"{not-json}\n")
    def test_repetition_is_warning_only_and_natural_stop_completes(self):
        phrase="repeated mathematical reasoning phrase with enough distinctive words"
        outputs=[];collector=DurableStreamCollector(self.path,warning_callback=outputs.append)
        repeated=" ".join([phrase]*30)
        result=collector.consume([self.chunk(repeated,done=True,done_reason="stop")],question_id="q",phase="initial",attempt_number=1,call_id="c")
        self.assertEqual(result["status"],"complete");self.assertTrue(any(x["diagnostic"]=="repetitive_output_warning" for x in outputs))
        self.assertTrue(any(x["event"]=="call_complete" for x in JsonlJournal.read_events(self.path)))
    def test_jsonl_corrupt_tail_is_detected_and_never_silently_repaired(self):
        self.path.write_bytes(b'{"event":"ok"}\n{"event":')
        with self.assertRaisesRegex(ValueError,"Incomplete JSONL tail"): JsonlJournal(self.path)
    def test_final_answer_parser_works_on_reconstructed_text(self):
        result=self.collector.consume([self.chunk("reason\n"),self.chunk("FINAL: -3/4",done=True,done_reason="stop")],question_id="q",phase="initial",attempt_number=1,call_id="c")
        import sys
        sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
        from run_math_study_reference import parse_final_answer
        self.assertEqual(parse_final_answer(result["content"]),("-3/4","parseable"))

if __name__=="__main__": unittest.main(verbosity=2)
