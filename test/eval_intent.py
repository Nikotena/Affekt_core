#!/usr/bin/env python3
"""
Intent/tone benchmark for free-text player input.

    python test/eval_intent.py            # current keyword system
    python test/eval_intent.py --errors   # list every wrong line

A classifier is any object with classify(text) -> (intent, valence). KeywordClassifier is
what KankeiEngine does today (narrate_action for the intent the LLM sees, and
_generate_action_delta for the emotional effect on the NPC).
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))

from modules.IntentParser import IntentParser  # noqa: E402
from player_lines import BASELINE_INTENTS, LINES  # noqa: E402
from player_lines_heldout import HELDOUT_LINES  # noqa: E402
from player_lines_stress import STRESS_LINES  # noqa: E402
from player_lines_user import USER_LINES  # noqa: E402

class KeywordClassifier:
    """The engine's ORIGINAL behaviour on free text, frozen here so the comparison stays
    reproducible now that the engine itself uses modules/IntentParser."""

    NARRATION = (
        (("attack", "fight"), "attack"), (("greet", "hello"), "greet"), (("help", "aid"), "offer_help"),
        (("buy", "purchase"), "buy"), (("sell", "trade"), "sell"), (("ask", "question"), "ask"),
    )

    @staticmethod
    def intent(text):
        low = text.lower()
        for keywords, intent in KeywordClassifier.NARRATION:
            if any(k in low for k in keywords):
                return intent
        return "other"

    @staticmethod
    def delta_x(text):
        low = text.lower()
        if "greet" in low:
            return 0.3
        if "fight" in low or "attack" in low:
            return -0.4
        if "ask" in low or "question" in low:
            return 0.2
        if "help" in low or "aid" in low:
            return 0.4
        return 0.1  # default small push

    def classify(self, text):
        x = self.delta_x(text)
        return self.intent(text), (1 if x > 0 else -1 if x < 0 else 0)


class RuleParserClassifier:
    """modules/IntentParser: the rule-based replacement."""

    def __init__(self):
        self._parser = IntentParser()

    def classify(self, text):
        p = self._parser.parse(text)
        return p.intent, p.valence


def evaluate(classifier, lines=LINES):
    rows = []
    for text, intent, valence in lines:
        got_intent, got_valence = classifier.classify(text)
        rows.append((text, intent, valence, got_intent, got_valence))
    return rows


def summarize(rows):
    n = len(rows)
    intent_ok = sum(r[1] == r[3] for r in rows)
    cover = [r for r in rows if r[1] in BASELINE_INTENTS]
    cover_ok = sum(r[1] == r[3] for r in cover)
    felt = [r for r in rows if r[2] != 0]          # lines that should provoke a feeling
    felt_ok = sum(r[2] == r[4] for r in felt)
    wrong_sign = sum(r[2] * r[4] < 0 for r in felt)
    neutral = [r for r in rows if r[2] == 0]
    false_feeling = sum(r[4] != 0 for r in neutral)
    by_intent = defaultdict(lambda: [0, 0])
    for r in rows:
        by_intent[r[1]][1] += 1
        by_intent[r[1]][0] += r[1] == r[3]
    return {
        "lines": n, "intent_acc": intent_ok / n,
        "intent_acc_coverable": cover_ok / len(cover), "coverable": len(cover),
        "valence_acc_felt": felt_ok / len(felt), "felt": len(felt),
        "opposite_feeling": wrong_sign, "false_feeling_on_neutral": false_feeling / len(neutral),
        "neutral": len(neutral), "by_intent": dict(by_intent),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--errors", action="store_true", help="print every line the classifier gets wrong")
    ap.add_argument("--classifier", choices=["keyword", "rules"], default="rules")
    ap.add_argument("--set", dest="which", choices=["dev", "heldout", "stress", "user", "all"], default="all")
    args = ap.parse_args()

    lines = {"dev": LINES, "heldout": HELDOUT_LINES, "stress": STRESS_LINES, "user": USER_LINES,
             "all": LINES + HELDOUT_LINES + STRESS_LINES + USER_LINES}[args.which]
    clf = KeywordClassifier() if args.classifier == "keyword" else RuleParserClassifier()
    rows = evaluate(clf, lines)
    s = summarize(rows)
    print(f"=== {args.classifier} on {s['lines']} free-text player lines ({args.which}) ===")
    print(f"intent, all lines:            {s['intent_acc']:.0%}")
    print(f"intent, labels it can output: {s['intent_acc_coverable']:.0%}  ({s['coverable']} lines)")
    print(f"feeling right (insult/threat/praise...): {s['valence_acc_felt']:.0%}  ({s['felt']} lines)")
    print(f"  of those, OPPOSITE feeling:  {s['opposite_feeling']}")
    print(f"neutral lines given a feeling anyway: {s['false_feeling_on_neutral']:.0%}  ({s['neutral']} lines)")
    print("\nper intent (right/total):")
    for intent, (ok, total) in sorted(s["by_intent"].items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {intent:11s} {ok}/{total}")
    if args.errors:
        print("\nwrong lines (expected -> got):")
        for text, intent, valence, gi, gv in rows:
            if intent != gi or valence != gv:
                print(f"  {text!r}: {intent}/{valence:+d} -> {gi}/{gv:+d}")


if __name__ == "__main__":
    main()
