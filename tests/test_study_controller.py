"""The run controller produces a handoff even when real execution is blocked."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class StudyController(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.study = self.project / "outputs/live_study/fixture"
        script = Path(__file__).resolve().parents[1] / "scripts/run_live_study.py"
        spec = importlib.util.spec_from_file_location("controller_fixture", script)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_preparation_failure_always_generates_reports_and_zip(self):
        with patch.object(self.module, "ROOT", self.project), patch.object(self.module, "execute", return_value=1) as call, \
             patch.object(sys, "argv", ["run_live_study.py", "--study-root", str(self.study), "--minutes", "1"]), contextlib.redirect_stdout(io.StringIO()):
            exit_code = self.module.main()
        self.assertEqual(exit_code, 2)
        self.assertEqual(call.call_args.args[2][0], "scripts/prepare_live_study.py")
        self.assertIn("--study-root", call.call_args.args[2])
        reports = self.project / "reports/live_study"
        self.assertTrue((reports / "CURRENT_STATUS.md").exists())
        self.assertTrue((reports / "CURRENT_STATUS.json").exists())
        self.assertTrue((reports / "diagnostic_bundle.zip").exists())
        status = json.loads((reports / "CURRENT_STATUS.json").read_text(encoding="utf-8"))
        self.assertEqual(status["overall_status"], "PARTIAL")
        self.assertEqual(status["execution"]["active_processes"], [])
        self.assertTrue(any("Pinned data preparation failed" in e.get("error", "") for e in status["errors"]))

    def test_collection_failure_still_invokes_guarded_training_then_handoff(self):
        self.study.mkdir(parents=True)
        (self.study / "questions.jsonl").write_text("", encoding="utf-8")
        (self.study / "protocol.json").write_text("{}", encoding="utf-8")
        with patch.object(self.module, "ROOT", self.project), patch.object(self.module, "execute", side_effect=[2, 0]) as call, \
             patch.object(sys, "argv", ["run_live_study.py", "--study-root", str(self.study), "--minutes", "1"]), contextlib.redirect_stdout(io.StringIO()):
            exit_code = self.module.main()
        self.assertEqual(exit_code, 2)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args_list[-1].args[2][0], "scripts/train_live_study.py")
        self.assertTrue((self.project / "reports/live_study/diagnostic_bundle.zip").exists())


if __name__ == "__main__":
    unittest.main()
