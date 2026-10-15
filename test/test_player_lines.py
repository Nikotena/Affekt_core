"""Sanity tests for the free-text player benchmark and its scoring."""
import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_intent import evaluate, summarize
from player_lines import INTENTS, LINES


class OracleClassifier:
    def __init__(self):
        self.answers = {t: (i, v) for t, i, v in LINES}

    def classify(self, text):
        return self.answers[text]


class AlwaysOther:
    def classify(self, text):
        return "other", 0


class PlayerLinesTests(unittest.TestCase):
    def test_data_is_clean(self):
        texts = [t for t, _i, _v in LINES]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertGreaterEqual(len(LINES), 60)
        counts = Counter(i for _t, i, _v in LINES)
        for intent in INTENTS:
            self.assertGreaterEqual(counts[intent], 3, intent)
        self.assertTrue(all(i in INTENTS and v in (-1, 0, 1) for _t, i, v in LINES))

    def test_valence_matches_intent(self):
        for text, intent, valence in LINES:
            if intent in ("insult", "threaten", "attack"):
                self.assertEqual(valence, -1, text)
            if intent in ("compliment", "offer_help", "farewell"):
                self.assertEqual(valence, 1, text)

    def test_scoring_with_perfect_and_trivial_classifiers(self):
        s = summarize(evaluate(OracleClassifier()))
        self.assertEqual((s["intent_acc"], s["valence_acc_felt"], s["opposite_feeling"]), (1.0, 1.0, 0))
        t = summarize(evaluate(AlwaysOther()))
        self.assertLess(t["intent_acc"], 0.2)
        self.assertEqual(t["false_feeling_on_neutral"], 0.0)


if __name__ == "__main__":
    unittest.main()
