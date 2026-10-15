#!/usr/bin/env python3
"""Score IntentParser on the hand-labelled lines (labels.json), split into dev and reserve.

    python test/eval_labeled.py --split          # one-time: write labels_dev.json / labels_reserve.json
    python test/eval_labeled.py --errors         # dev accuracy + the dev lines it gets wrong
    python test/eval_labeled.py --reserve        # reserve accuracy only (no lines shown)
    python test/eval_labeled.py --llm [--reserve]   # same, with the AMD API model resolving "other"

Tune the rules on dev. The reserve lines are never printed, so their accuracy stays an honest
estimate of how the parser does on lines nobody looked at while writing rules.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules.IntentParser import parse  # noqa: E402

DATA = os.path.join(ROOT, "test", "data")
ALL, DEV, RESERVE = (os.path.join(DATA, n) for n in ("labels.json", "labels_dev.json", "labels_reserve.json"))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split(n_reserve=50, seed=42):
    rows = load(ALL)
    random.Random(seed).shuffle(rows)
    for path, part in ((RESERVE, rows[:n_reserve]), (DEV, rows[n_reserve:])):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=1)
    print(f"dev {len(rows) - n_reserve}, reserve {n_reserve}")


def llm_intents(rows, workers=6):
    """Intent per row after the model resolves the parser's "other" answers (one call each)."""
    from concurrent.futures import ThreadPoolExecutor
    from modules.IntentLLM import refine
    from modules.OpenRouterBackend import AMDBackend
    backend = AMDBackend(temperature=0.0, timeout=60.0)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [p.intent for p in pool.map(lambda r: refine(parse(r["text"]), backend), rows)]


def score(rows, show=False, intents=None):
    right, conf = 0, Counter()
    for i, r in enumerate(rows):
        got = intents[i] if intents else parse(r["text"]).intent
        if got == r["intent"]:
            right += 1
        else:
            conf[(r["intent"], got)] += 1
            if show:
                print(f"  want {r['intent']:<10} got {got:<10} {r['text']}")
    return right, len(rows), conf


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--reserve", action="store_true")
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    if a.split:
        return split()
    path, name = (RESERVE, "reserve") if a.reserve else (DEV, "dev")
    rows = load(path)
    right, n, conf = score(rows, show=a.errors and not a.reserve)
    print(f"{name}: rules only      {right}/{n} ({100 * right / n:.0f}%)")
    if a.llm:
        asked = sum(parse(r["text"]).intent == "other" for r in rows)
        right, n, conf = score(rows, show=a.errors and not a.reserve, intents=llm_intents(rows))
        print(f"{name}: rules + model   {right}/{n} ({100 * right / n:.0f}%)  [{asked} model calls]")
    if not a.reserve:
        print("confusions:", ", ".join(f"{w}->{g} x{c}" for (w, g), c in conf.most_common(8)))


if __name__ == "__main__":
    main()
