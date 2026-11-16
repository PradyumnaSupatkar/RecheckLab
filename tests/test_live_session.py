"""Prevent duplicate live calls across repeated clicks and overlapping reruns."""
import copy
from collections.abc import MutableMapping
from threading import Event, Thread
import unittest

from rechecklab.live_session import execute_once, live_input_key


def backend():
    return dict(model="test-only-solver", model_digest="test-only-digest",
                protocol_fingerprint="test-only-protocol",
                generation_settings=dict(budget=768, seed=42, temperature=0.0, thinking=False))


class LiveSessionSafeguards(unittest.TestCase):
    def test_input_key_is_stable_for_equivalent_context(self):
        status = backend()
        key = live_input_key("User supplied question", "model-artifact-a", "http://localhost:11434", status)
        reordered = dict(reversed(list(status.items())))
        reordered["generation_settings"] = dict(reversed(list(status["generation_settings"].items())))
        self.assertEqual(key, live_input_key("  User supplied question\n", "model-artifact-a",
                                             "http://localhost:11434/", reordered))
        self.assertEqual(status, backend())

    def test_changed_question_artifact_endpoint_solver_digest_or_protocol_invalidates_key(self):
        base = backend()
        original = live_input_key("Question A", "artifact-a", "http://localhost:11434", base)
        variations = [
            ("Question B", "artifact-a", "http://localhost:11434", base),
            ("Question A", "artifact-b", "http://localhost:11434", base),
            ("Question A", "artifact-a", "http://127.0.0.1:11434", base),
        ]
        for field, changed in (("model", "different-solver"), ("model_digest", "different-checkpoint"),
                               ("protocol_fingerprint", "different-protocol")):
            variations.append(("Question A", "artifact-a", "http://localhost:11434", {**base, field: changed}))
        changed_settings = copy.deepcopy(base)
        changed_settings["generation_settings"]["budget"] = 4096
        variations.append(("Question A", "artifact-a", "http://localhost:11434", changed_settings))
        for index, args in enumerate(variations):
            with self.subTest(context=index):
                self.assertNotEqual(original, live_input_key(*args))

    def test_same_request_returns_cached_result_and_never_repeats_operation(self):
        state, calls = {}, []
        result = dict(status="complete", decision="Skip recheck", selected_answer="test-fixture-only")

        def operation():
            calls.append("called")
            return result

        first = execute_once(state, "request-a", operation)
        second = execute_once(state, "request-a", operation)
        self.assertEqual(first, dict(status="complete", result=result))
        self.assertIs(first, second)
        self.assertEqual(calls, ["called"])

    def test_failed_exception_is_cached_and_lock_is_released(self):
        state, calls = {}, []

        def failure():
            calls.append("failed-attempt")
            raise RuntimeError("test-only returned partial response")

        first = execute_once(state, "failed-request", failure)
        self.assertEqual(first["status"], "failed")
        self.assertIn("test-only returned partial response", first["error"])
        self.assertIs(first, execute_once(state, "failed-request", failure))
        self.assertEqual(calls, ["failed-attempt"])
        next_job = execute_once(state, "different-request", lambda: "successful-next-attempt")
        self.assertEqual(next_job, dict(status="complete", result="successful-next-attempt"))

    def test_returned_engine_failure_is_cached_as_an_attempt(self):
        state, calls = {}, []
        failure = dict(status="recheck_failed", initial=dict(answer="retained test-only answer"),
                       rechecked=False, error="test-only unavailable revision")

        def operation():
            calls.append("initial-and-recheck-attempt")
            return failure

        first = execute_once(state, "request-with-failed-recheck", operation)
        self.assertEqual(first["result"], failure)
        self.assertIs(first, execute_once(state, "request-with-failed-recheck", operation))
        self.assertEqual(len(calls), 1)

    def test_interruption_after_request_sent_retains_tombstone_without_repeating_call(self):
        class ScriptInterrupted(BaseException):
            """Model a Streamlit stop/rerun escaping normal Exception handling."""

        state, sent = {}, []

        def interrupted_operation():
            sent.append("HTTP request sent")
            raise ScriptInterrupted("test-only interrupted script")

        with self.assertRaises(ScriptInterrupted):
            execute_once(state, "interrupted-request", interrupted_operation)
        self.assertEqual(state["_live_jobs"]["interrupted-request"], dict(status="busy"))
        self.assertEqual(execute_once(state, "interrupted-request", interrupted_operation), dict(status="busy"))
        self.assertEqual(sent, ["HTTP request sent"])
        # The interrupted key stays protected, while the session lock permits
        # an explicitly different request after the original script exits.
        self.assertEqual(execute_once(state, "different-request", lambda: "different-result"),
                         dict(status="complete", result="different-result"))

    def test_stopped_streamlit_adapter_releases_lock_without_session_state_cleanup(self):
        from streamlit.runtime.state.safe_session_state import SafeSessionState
        from streamlit.runtime.state.session_state import SessionState
        from streamlit.runtime.scriptrunner_utils.exceptions import StopException

        class StateAdapter(MutableMapping):
            """Match the proxy's mapping API while retaining real yield checks."""
            def __init__(self, safe_state):
                self.safe_state = safe_state

            def __getitem__(self, key):
                return self.safe_state[key]

            def __setitem__(self, key, value):
                self.safe_state[key] = value

            def __delitem__(self, key):
                del self.safe_state[key]

            def __iter__(self):
                return iter(self.safe_state.filtered_state)

            def __len__(self):
                return len(self.safe_state.filtered_state)

        underlying = SessionState()
        stopped, sent = [False], []

        def old_runner_yield():
            if stopped[0]:
                raise StopException()

        old_state = StateAdapter(SafeSessionState(underlying, old_runner_yield))

        def interrupted_operation():
            sent.append("HTTP request sent")
            stopped[0] = True
            # A progress update or state access observes the runner's STOP.
            old_runner_yield()

        with self.assertRaises(StopException):
            execute_once(old_state, "interrupted-request", interrupted_operation)
        # A stopped runner cannot reset a scalar busy flag in a finally block.
        # The implementation releases its local lock without such a mutation.
        with self.assertRaises(StopException):
            old_state["cleanup_marker"] = "would never be written"

        new_state = StateAdapter(SafeSessionState(underlying, lambda: None))
        self.assertFalse(new_state["_live_lock"].locked())
        self.assertNotIn("cleanup_marker", new_state)
        self.assertEqual(execute_once(new_state, "interrupted-request", interrupted_operation), dict(status="busy"))
        self.assertEqual(sent, ["HTTP request sent"])
        self.assertEqual(execute_once(new_state, "new-question", lambda: "new-result"),
                         dict(status="complete", result="new-result"))

    def test_overlapping_same_or_distinct_request_is_busy_until_first_finishes(self):
        state, calls, completed = {}, [], []
        started, release = Event(), Event()

        def first_operation():
            calls.append("first")
            started.set()
            if not release.wait(2):
                raise RuntimeError("Test did not release its bounded worker.")
            return "first-result"

        def duplicate_operation():
            calls.append("unwanted-duplicate")
            return "duplicate-result"

        worker = Thread(target=lambda: completed.append(execute_once(state, "first-request", first_operation)), daemon=True)
        worker.start()
        try:
            self.assertTrue(started.wait(1), "Worker never entered its operation.")
            self.assertEqual(execute_once(state, "first-request", duplicate_operation), dict(status="busy"))
            self.assertEqual(execute_once(state, "different-request", duplicate_operation), dict(status="busy"))
            self.assertEqual(calls, ["first"])
        finally:
            release.set()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive(), "Worker did not finish within the bounded timeout.")
        self.assertEqual(completed, [dict(status="complete", result="first-result")])
        self.assertIs(completed[0], execute_once(state, "first-request", duplicate_operation))
        self.assertEqual(execute_once(state, "different-request", lambda: "second-result"),
                         dict(status="complete", result="second-result"))

    def test_clearing_display_fields_preserves_jobs_lock_and_previous_requests(self):
        state = dict(live_result="old display", live_error="old error", live_request_key="old key")
        calls = []
        first = execute_once(state, "request-0", lambda: calls.append("first") or "first-result")
        lock = state["_live_lock"]
        for field in ("live_result", "live_error", "live_request_key"):
            state.pop(field, None)
        for index in range(1, 8):
            execute_once(state, f"request-{index}", lambda index=index: f"result-{index}")
        self.assertIs(lock, state["_live_lock"])
        self.assertEqual(len(state["_live_jobs"]), 8)
        self.assertIs(first, execute_once(state, "request-0", lambda: calls.append("duplicate")))
        self.assertEqual(calls, ["first"])


if __name__ == "__main__":
    unittest.main()
