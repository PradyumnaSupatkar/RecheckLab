"""Mathematical precision checks; fixtures are never collection evidence."""
from decimal import Decimal, getcontext, localcontext
import json
from pathlib import Path
import unittest

from rechecklab.study_protocol import load_protocol, normalize_number
from rechecklab.study_validation import audit_numeric_precision
from test_study_collection import fixture_protocol, generated


class FrozenNumericPrecision(unittest.TestCase):
    def setUp(self):
        self.protocol = fixture_protocol()

    def audit(self, answer):
        return audit_numeric_precision(self.protocol, generated(answer), "number")

    def test_wrong_overprecision_answer_cannot_round_to_a_correct_integer(self):
        wrong = "1.00000000000000000000000000001"
        self.assertEqual(normalize_number(wrong), "1")
        self.assertNotEqual(Decimal(wrong), Decimal("1"))
        with self.assertRaisesRegex(ValueError, "precision loss"):
            self.audit(wrong)

    def test_exact_integers_decimal_scientific_currency_and_thousands_are_preserved(self):
        for answer in ("0", "102", "-35", "₹1,200.00", "0.000125", "-2.5e2",
                       "1234567890123456789012345678", "100000000000000000000000000000000000000"):
            with self.subTest(answer=answer):
                result = self.audit(answer)
                self.assertEqual(result["status"], "EXACT")
                self.assertTrue(result["exact_value_verified"])
                self.assertEqual(result["numeric_form"], "decimal")

    def test_fraction_requires_exact_finite_representation_within_frozen_precision(self):
        for answer in ("1/2", "1/8", "-25/100", "2/2", "0/3", "1/1099511627776",
                       "1/1000000000000000000000000000000"):
            with self.subTest(answer=answer):
                self.assertEqual(self.audit(answer)["status"], "EXACT")
        for answer in ("1/3", "2/7", "1/2199023255552", "100000000000000000000000000001/1"):
            with self.subTest(answer=answer), self.assertRaisesRegex(ValueError, "precision loss"):
                self.audit(answer)

    def test_unparseable_answer_stays_unparseable_without_fabricating_a_number(self):
        for answer in ("NaN", "1/0", "1,5", "2 or 3"):
            with self.subTest(answer=answer):
                evidence = self.audit(answer)
                self.assertEqual(evidence["status"], "UNPARSEABLE")
                self.assertFalse(evidence["exact_value_verified"])
        self.assertEqual(audit_numeric_precision(self.protocol, generated("A"), "choice")["status"], "NOT_APPLICABLE")

    def test_audit_preserves_caller_context_and_still_checks_frozen_precision(self):
        original = getcontext().copy()
        with localcontext() as context:
            context.prec = 100
            with self.assertRaisesRegex(ValueError, "precision loss"):
                self.audit("1.00000000000000000000000000001")
            self.assertEqual(context.prec, 100)
        self.assertEqual(getcontext().prec, original.prec)

    def test_last_explicit_marker_only_and_actual_content_are_required(self):
        response = generated("1")
        response["message"]["content"] = "FINAL: 1.00000000000000000000000000001\nCorrection\nFINAL: 2"
        self.assertEqual(audit_numeric_precision(self.protocol, response, "number")["status"], "EXACT")
        with self.assertRaisesRegex(ValueError, "actual saved response"):
            audit_numeric_precision(self.protocol, {"output": {"answer": "2"}}, "number")

    def test_available_genuine_numeric_calls_preserve_the_exact_marked_value(self):
        root = Path(__file__).resolve().parents[1] / "outputs" / "live_study" / "v0.3.0"
        if not (root / "calls.jsonl").is_file() or not (root / "protocol.json").is_file():
            self.skipTest("No separately collected local-study journal is available.")
        protocol, _ = load_protocol(root)
        questions = {q["example_id"]: q for q in (json.loads(line) for line in
                     (root / "questions.jsonl").read_text(encoding="utf-8").split("\n") if line.strip())}
        snapshot = (root / "calls.jsonl").read_bytes()
        lines = snapshot.decode("utf-8").split("\n")
        if snapshot and not snapshot.endswith(b"\n"):
            lines = lines[:-1]  # A concurrently appended fragment is not completed evidence.
        checked = 0
        for event in (json.loads(line) for line in lines if line.strip()):
            if event.get("status") != "succeeded" or questions[event["example_id"]]["answer_type"] != "number":
                continue
            with self.subTest(call_id=event["call_id"]):
                result = audit_numeric_precision(protocol, event["response"], "number")
                self.assertIn(result["status"], ("EXACT", "UNPARSEABLE"))
                checked += 1
        if not checked:
            self.skipTest("No completed numeric calls are available in this snapshot.")


if __name__ == "__main__":
    unittest.main()
