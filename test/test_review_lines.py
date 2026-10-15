"""The batch review workflow for the player's own lines."""
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import review_lines as rl


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def write(self, name, text):
        with open(self.path(name), "w", encoding="utf-8") as f:
            f.write(text)
        return self.path(name)

    def test_reads_numbered_lines_skips_blanks_and_duplicates(self):
        p = self.write("t.txt", "1. Hello there!\n\n2) hello there!\n3. Where is the inn?\n   \n10. Die, you wretch!\n")
        self.assertEqual(rl.read_lines(p), ["Hello there!", "Where is the inn?", "Die, you wretch!"])

    def test_csv_roundtrip_and_scoring(self):
        lines = ["Hello there!", "Where is the inn?", "Die, you wretch!", "Your mother was a hamster."]
        out = self.path("r.csv")
        rl.write_review(lines, out)
        with open(out, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["text"] for r in rows], lines)
        self.assertTrue(all(r["fix_intent"] == "" and r["fix_valence"] == "" for r in rows))

        s = rl.score(out)                                  # nothing corrected: the parser is "right" by default
        self.assertEqual((s["n"], s["intent_acc"]), (4, 1.0))

        rows[3]["fix_intent"] = "insult"                   # the person fixes the one it missed
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=rl.COLUMNS)
            w.writeheader()
            w.writerows(rows)
        labels = self.path("labels.json")
        s = rl.score(out, labels)
        self.assertEqual((s["n"], s["intent_acc"], s["feeling_acc"], s["opposite"]), (4, 0.75, 0.75, 0))  # missed, not reversed
        import json
        data = json.load(open(labels, encoding="utf-8"))
        self.assertEqual(data[3], {"text": lines[3], "intent": "insult", "valence": -1})

    def test_bad_fix_values_are_reported_not_crashed(self):
        out = self.path("r.csv")
        rl.write_review(["Hello there!", "Where is the inn?"], out)
        with open(out, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        rows[0]["fix_intent"] = "grumpy"
        rows[1]["fix_valence"] = "maybe"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=rl.COLUMNS)
            w.writeheader()
            w.writerows(rows)
        s = rl.score(out)
        self.assertEqual(len(s["problems"]), 2)
        self.assertEqual(s["n"], 0)

    def test_valence_of_intent(self):
        self.assertEqual([rl.valence_of(i) for i in ("compliment", "insult", "ask")], [1, -1, 0])


if __name__ == "__main__":
    unittest.main()
