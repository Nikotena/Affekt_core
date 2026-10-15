"""Regression floor for IntentParser on Aaron's hand-labelled lines (test/data/labels_*.json)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))

import eval_labeled as ev  # noqa: E402
from modules.IntentParser import parse  # noqa: E402


class LabeledLinesTests(unittest.TestCase):
    def test_dev_accuracy_does_not_drop(self):
        right, n, _ = ev.score(ev.load(ev.DEV))
        self.assertGreaterEqual(right / n, 0.65)  # 45% before the rules were tuned on these lines

    def test_reserve_accuracy_does_not_drop(self):
        right, n, _ = ev.score(ev.load(ev.RESERVE))
        self.assertGreaterEqual(right / n, 0.50)  # lines never used for tuning: 52% before

    def test_split_is_complete_and_disjoint(self):
        dev, reserve = ev.load(ev.DEV), ev.load(ev.RESERVE)
        self.assertEqual((len(dev), len(reserve)), (150, 50))
        self.assertFalse({r["text"] for r in dev} & {r["text"] for r in reserve})

    def test_not_violence_and_not_a_purchase(self):
        self.assertNotEqual(parse("I charge by the minute, so make it quick.").intent, "attack")
        self.assertNotEqual(parse("Give me a second.").intent, "buy")
        self.assertNotEqual(parse("I'll take the job.").intent, "buy")


if __name__ == "__main__":
    unittest.main()
