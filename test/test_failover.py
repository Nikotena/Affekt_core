"""AMD first, another provider as a backup: order, cooldown, empty replies and the provider label."""
import unittest

import modules.ApiBackend as orb
from modules.ApiBackend import APIError, FailoverBackend, describe
from modules.LLMLayer import LLMBackend

PAYLOAD = ["You are Marta, the Merchant.", "Style: neutral and businesslike."]


class Fake(LLMBackend):
    provider = "Fake"

    def __init__(self, model, replies):
        self.model = model
        self.replies = list(replies)  # a str is returned, an Exception is raised
        self.calls = 0
        self.shots = 3

    def generate(self, payload, action):
        self.calls += 1
        r = self.replies.pop(0) if self.replies else self.replies_default
        if isinstance(r, Exception):
            raise r
        return r

    replies_default = "ok"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FailoverTests(unittest.TestCase):
    def make(self, a, b, cooldown=45.0):
        clock = Clock()
        return FailoverBackend([a, b], cooldown=cooldown, clock=clock), clock

    def test_primary_answers_and_backup_is_not_called(self):
        a, b = Fake("amd", ["Hello."]), Fake("or", ["Backup."])
        fo, _ = self.make(a, b)
        self.assertEqual(fo.generate(PAYLOAD, "hi"), "Hello.")
        self.assertEqual((a.calls, b.calls), (1, 0))
        self.assertEqual(fo.last_provider(), "Fake · amd")

    def test_backup_answers_when_primary_fails(self):
        a, b = Fake("amd", [APIError("HTTP 429")]), Fake("or", ["Backup."])
        fo, _ = self.make(a, b)
        self.assertEqual(fo.generate(PAYLOAD, "hi"), "Backup.")
        self.assertEqual(fo.last_provider(), "Fake · or")

    def test_failed_primary_is_skipped_during_the_cooldown_then_retried(self):
        a, b = Fake("amd", [APIError("timeout"), "Back on AMD."]), Fake("or", ["B1", "B2", "B3"])
        fo, clock = self.make(a, b, cooldown=45.0)
        self.assertEqual(fo.generate(PAYLOAD, "1"), "B1")
        self.assertEqual(fo.generate(PAYLOAD, "2"), "B2")  # AMD is cooling down: straight to the backup
        self.assertEqual(a.calls, 1)
        clock.t += 46
        self.assertEqual(fo.generate(PAYLOAD, "3"), "Back on AMD.")  # cooldown over: AMD gets another chance
        self.assertEqual(fo.last_provider(), "Fake · amd")

    def test_empty_reply_moves_on_once_without_a_cooldown(self):
        a, b = Fake("amd", ["  ", "Real answer."]), Fake("or", ["Backup."])
        fo, _ = self.make(a, b)
        self.assertEqual(fo.generate(PAYLOAD, "1"), "Backup.")
        self.assertEqual(fo.generate(PAYLOAD, "2"), "Real answer.")  # AMD was not punished for one empty reply

    def test_all_failing_raises_so_the_engine_uses_its_canned_line(self):
        a, b = Fake("amd", [APIError("429")]), Fake("or", [APIError("503")])
        fo, _ = self.make(a, b)
        with self.assertRaises(APIError):
            fo.generate(PAYLOAD, "hi")
        self.assertIsNone(fo.last_provider())

    def test_everything_cooling_down_still_tries_everyone(self):
        a = Fake("amd", [APIError("x"), "A again"])
        b = Fake("or", [APIError("y")])
        fo, _ = self.make(a, b)
        with self.assertRaises(APIError):
            fo.generate(PAYLOAD, "1")
        self.assertEqual(fo.generate(PAYLOAD, "2"), "A again")  # both were down, the order was kept

    def test_shots_are_shared_by_all_backends(self):
        a, b = Fake("amd", []), Fake("or", [])
        fo, _ = self.make(a, b)
        fo.shots = 2
        self.assertEqual((a.shots, b.shots, fo.shots), (2, 2, 2))

    def test_clear_last_forgets_the_previous_answer(self):
        a, b = Fake("amd", ["Hi."]), Fake("or", [])
        fo, _ = self.make(a, b)
        fo.generate(PAYLOAD, "hi")
        fo.clear_last()
        self.assertIsNone(fo.last_provider())

    def test_needs_at_least_one_backend(self):
        with self.assertRaises(ValueError):
            FailoverBackend([])

    def test_describe_names_provider_and_model(self):
        b = orb.AMDBackend(api_key="k")
        self.assertEqual(describe(b), f"AMD Token Factory · {orb.AMD_DEFAULT_MODEL}")
        self.assertEqual(describe(orb.AMDBackend(model="Gemma-4-31B-IT", api_key="k")),
                         "AMD Token Factory · Gemma-4-31B-IT")

    def test_shared_state_remembers_a_failure_between_instances(self):
        # the game builds one FailoverBackend per request; a shared dict keeps the cooldown across them
        state, clock = {}, Clock()
        a1, b1 = Fake("amd", [APIError("HTTP 503")]), Fake("or", ["Backup."])
        FailoverBackend([a1, b1], cooldown=45.0, clock=clock, state=state).generate(PAYLOAD, "hi")
        a2, b2 = Fake("amd", ["Hello."]), Fake("or", ["Backup 2."])
        out = FailoverBackend([a2, b2], cooldown=45.0, clock=clock, state=state).generate(PAYLOAD, "hi")
        self.assertEqual((a2.calls, out), (0, "Backup 2."))
        clock.t += 46
        a3, b3 = Fake("amd", ["Back."]), Fake("or", ["x"])
        self.assertEqual(FailoverBackend([a3, b3], cooldown=45.0, clock=clock, state=state).generate(PAYLOAD, "hi"), "Back.")
        self.assertNotIn("amd", state)
        self.assertTrue(all(isinstance(k, str) for k in state))  # keyed by model name, never by an API key


if __name__ == "__main__":
    unittest.main()
