#!/usr/bin/env python3
"""
Laya (post-hackathon experiment) on the same intent/tone benchmark as the rules.

Needs the `laya` package; run it with a Python that has it, e.g. the isolated venv:
    USE_TF=0 <venv>/Scripts/python.exe test/eval_laya.py --set all [--errors]

It asks two typed questions per line (intent: choice, feeling: choice) in one forward pass.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))

from eval_intent import evaluate, summarize  # noqa: E402
from player_lines import LINES  # noqa: E402
from player_lines_heldout import HELDOUT_LINES  # noqa: E402
from player_lines_stress import STRESS_LINES  # noqa: E402
from player_lines_user import USER_LINES  # noqa: E402

INTENT_CRITERIA = {
    "greet": "says hello or greets the character",
    "farewell": "says goodbye or announces leaving",
    "ask": "asks a question or asks for information",
    "ask_help": "asks the character for help, protection or rescue",
    "offer_help": "offers to help the character",
    "buy": "wants to buy something",
    "haggle": "argues about the price or offers a different price",
    "sell": "wants to sell or trade something to the character",
    "compliment": "praises or thanks the character",
    "insult": "insults or belittles the character",
    "complain": "complains, is frustrated or impatient",
    "threaten": "threatens the character",
    "attack": "physically attacks the character",
    "other": "none of these, small talk or a comment",
}
FEELING_CRITERIA = {
    "hostile": "angry, rude, threatening or aggressive towards the character",
    "neutral": "neutral, factual or businesslike",
    "pleasant": "friendly, kind, grateful or complimentary towards the character",
}
QUESTIONS = {
    "intent": {"type": "choice", "instructions": "What does the player want to do?", "criteria": INTENT_CRITERIA},
    "feeling": {"type": "choice", "instructions": "How does the player feel towards the character?",
                "criteria": FEELING_CRITERIA},
}
FEELING_TO_VALENCE = {"hostile": -1, "neutral": 0, "pleasant": 1}


class LayaClassifier:
    def __init__(self, checkpoint="convaiinnovations/laya-typed-decisions"):
        import laya
        self._agent = laya.load(checkpoint)
        self.seconds = 0.0
        self.calls = 0

    def classify(self, text):
        t = time.time()
        r = self._agent.predict("A player says to a character in a fantasy game: " + text, QUESTIONS)
        self.seconds += time.time() - t
        self.calls += 1
        a = r["answers"]
        return a["intent"]["choice"], FEELING_TO_VALENCE[a["feeling"]["choice"]]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="which", choices=["dev", "heldout", "stress", "user", "all"], default="all")
    ap.add_argument("--errors", action="store_true")
    args = ap.parse_args()
    sets = {"dev": LINES, "heldout": HELDOUT_LINES, "stress": STRESS_LINES, "user": USER_LINES}
    clf = LayaClassifier()
    for name in (sets if args.which == "all" else [args.which]):
        rows = evaluate(clf, sets[name])
        s = summarize(rows)
        print(f"{name:8s} n={s['lines']:3d} intent {s['intent_acc']:.0%}  feeling {s['valence_acc_felt']:.0%} "
              f"opposite {s['opposite_feeling']}  neutral-with-feeling {s['false_feeling_on_neutral']:.0%}")
        if args.errors:
            for text, intent, valence, gi, gv in rows:
                if intent != gi or valence != gv:
                    print(f"     {text!r}: {intent}/{valence:+d} -> {gi}/{gv:+d}")
    print(f"avg {1000 * clf.seconds / clf.calls:.0f} ms per line ({clf.calls} lines)")


if __name__ == "__main__":
    main()
