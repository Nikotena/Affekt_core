"""Prompt-cache warm-up: every prompt prefix the rotation can produce is read once at startup."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import KankeiEngine as ke
from modules.LLMLayer import (MOOD_STYLE, LlamaCppBackend, StubBackend, build_messages, example_windows,
                              select_few_shot, set_example_seed, set_example_variants)


class WindowTests(unittest.TestCase):
    def tearDown(self):
        set_example_variants(0)

    def test_windows_are_exactly_what_the_rotation_can_show(self):
        set_example_variants(5)
        set_example_seed(1)
        for mood in MOOD_STYLE:
            windows = example_windows(mood, 2)
            seen = {repr(select_few_shot(mood, limit=2)) for _ in range(200)}
            self.assertEqual({repr(w) for w in windows}, seen, mood)
            self.assertTrue(all(len(w) == 2 for w in windows))

    def test_every_set_is_different_up_to_the_pool_size(self):
        from modules.LLMLayer import _EXAMPLES
        set_example_variants(5)
        for mood in MOOD_STYLE:
            pool = len([e for e in _EXAMPLES if e[0] == mood])
            windows = example_windows(mood, 2)
            self.assertEqual(len({repr(w) for w in windows}), min(5, pool), mood)

    def test_nothing_to_warm_when_rotation_is_off_or_no_shots(self):
        self.assertEqual(example_windows("hostile", 2), [])
        set_example_variants(5)
        self.assertEqual(example_windows("hostile", 0), [])

    def test_explicit_examples_are_used_verbatim(self):
        set_example_variants(5)
        window = example_windows("hostile", 2)[3]
        msgs = build_messages(["You are X.", "Style: y."], "Hello there!", mood="hostile", shots=2, examples=window)
        replies = [m["content"] for m in msgs if m["role"] == "assistant"]
        self.assertEqual(replies, [reply for _f, _a, reply in window])
        self.assertEqual(msgs[-1]["role"], "user")


class BackendWarmUpTests(unittest.TestCase):
    def tearDown(self):
        set_example_variants(0)

    def backend(self):
        b = LlamaCppBackend.__new__(LlamaCppBackend)  # no model: only the prompt logic is under test
        b.shots, b._is_qwen, b.calls = 2, False, []
        b._complete = lambda messages, max_tokens: b.calls.append((messages, max_tokens)) or ""
        return b

    def test_every_mood_and_example_set_is_read_once_with_one_token(self):
        set_example_variants(5)
        b = self.backend()
        stats = b.warm_up()
        expected = sum(len(example_windows(m, 2)) for m in MOOD_STYLE)
        self.assertEqual(stats["prefixes"], expected)
        self.assertEqual(len(b.calls), expected)
        self.assertTrue(all(tokens == 1 for _m, tokens in b.calls))
        prefixes = {repr(m[:-1]) for m, _t in b.calls}       # everything before the real situation
        self.assertEqual(len(prefixes), expected)             # all different: no wasted work
        self.assertTrue(all(m[0]["role"] == "system" and m[-1]["role"] == "user" for m, _t in b.calls))

    def test_warm_prefix_matches_the_prefix_of_real_prompts(self):
        set_example_variants(5)
        b = self.backend()
        b.warm_up(moods=["hostile"])
        warm_prefixes = {repr(m[:-1]) for m, _t in b.calls}
        set_example_seed(2)
        real = {repr(build_messages(["You are Marta, the Merchant.", "Style: openly hostile, rude and menacing; "
                                     "insult or threaten the player; never polite."], "attack the NPC", shots=2,
                                    no_think=False)[:-1])
                for _ in range(60)}
        self.assertTrue(real <= warm_prefixes)

    def test_only_the_requested_moods(self):
        set_example_variants(5)
        b = self.backend()
        self.assertEqual(b.warm_up(moods=["tense"])["prefixes"], len(example_windows("tense", 2)))

    def test_rotation_off_means_nothing_to_do(self):
        b = self.backend()
        self.assertEqual(b.warm_up()["prefixes"], 0)
        self.assertEqual(b.calls, [])

    def test_stub_backend_has_a_noop_warm_up(self):
        self.assertEqual(StubBackend().warm_up(), {"prefixes": 0, "seconds": 0.0})


class EngineWarmUpTests(unittest.TestCase):
    class Recorder(StubBackend):
        def __init__(self):
            self.calls = 0

        def warm_up(self, moods=None):
            self.calls += 1
            return {"prefixes": 18, "seconds": 1.5}

    def test_create_engine_warms_up_only_when_asked(self):
        rec = self.Recorder()
        e = ke.create_engine(llm=rec)
        self.assertEqual((rec.calls, e.warm_up_stats), (0, None))
        rec2 = self.Recorder()
        e2 = ke.create_engine(llm=rec2, warm_up=True)
        self.assertEqual((rec2.calls, e2.warm_up_stats), (1, {"prefixes": 18, "seconds": 1.5}))


if __name__ == "__main__":
    unittest.main()
