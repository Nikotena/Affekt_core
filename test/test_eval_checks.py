"""Tests for the eval's leak/variety checks and the few-shot rotation."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_llm import leaks_example
from modules.LLMLayer import _EXAMPLES, MOOD_STYLE, narrate_action, select_few_shot, set_example_seed


class LeakTests(unittest.TestCase):
    def test_detects_reused_phrase(self):
        self.assertTrue(leaks_example("Now keep walking. I dislike your face, friend."))
        self.assertTrue(leaks_example("Then grab your rope and vanish, thief."))

    def test_ignores_generic_phrases(self):
        self.assertFalse(leaks_example("What do you want from me?"))
        self.assertFalse(leaks_example("Nothing like this has happened before."))


class RotationTests(unittest.TestCase):
    def test_every_mood_has_enough_examples(self):
        for mood in MOOD_STYLE:
            self.assertGreaterEqual(len([e for e in _EXAMPLES if e[0] == mood]), 2, mood)

    def test_seed_is_reproducible_and_varies(self):
        set_example_seed(1)
        a = [select_few_shot("hostile") for _ in range(5)]
        set_example_seed(1)
        b = [select_few_shot("hostile") for _ in range(5)]
        self.assertEqual(a, b)
        self.assertGreater(len({repr(x) for x in a}), 1)  # not always the same pair
        set_example_seed(None)

    def test_one_example_matches_the_action(self):
        wanted = narrate_action("buy a potion from me")
        for mood in ("friendly", "hostile", "tense", "neutral"):
            for seed in range(5):
                set_example_seed(seed)
                shots = select_few_shot(mood, action=wanted)
                self.assertEqual(len(shots), 3)
                self.assertIn(wanted, [a for _f, a, _r in shots], mood)
        set_example_seed(None)

    def test_examples_do_not_contain_real_game_prices_or_items(self):
        text = " ".join(" ".join(f) + r for _m, f, _a, r in _EXAMPLES).lower()
        for word in ("potion", "sword", "25", "100", "tavern", "market"):
            self.assertNotIn(word, text)

    def test_example_prices_are_all_different(self):
        prices = [w for _m, f, _a, _r in _EXAMPLES for line in f for w in line.split() if w.isdigit()]
        self.assertEqual(len(prices), len(set(prices)))

    def test_neutral_lines_are_not_tiny(self):
        for m, _f, _a, reply in _EXAMPLES:
            if m == "neutral":
                self.assertGreaterEqual(len(reply.split()), 8, reply)

    def test_examples_carry_the_style_line(self):
        for facts, _action, _reply in select_few_shot("tense"):
            self.assertTrue(any(f.startswith("Style: ") for f in facts))


if __name__ == "__main__":
    unittest.main()
