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
    from modules.ApiBackend import AMDBackend
    backend = AMDBackend(temperature=0.0, timeout=60.0)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [p.intent for p in pool.map(lambda r: refine(parse(r["text"]), backend), rows)]


def wilson(right, n, z=1.96):
    """95% Wilson confidence interval for a proportion, as (low, high)."""
    if not n:
        return 0.0, 0.0
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return centre - half, centre + half


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
        rules_right = right
        asked = sum(parse(r["text"]).intent == "other" for r in rows)
        with_model = llm_intents(rows)
        right, n, conf = score(rows, show=a.errors and not a.reserve, intents=with_model)
        print(f"{name}: rules + model   {right}/{n} ({100 * right / n:.0f}%)  [{asked} model calls]")
        # paired view: which individual lines the model fixed or broke (never prints the text)
        fixed = sum(1 for r, m in zip(rows, with_model) if m == r["intent"] and parse(r["text"]).intent != r["intent"])
        broke = sum(1 for r, m in zip(rows, with_model) if m != r["intent"] and parse(r["text"]).intent == r["intent"])
        lo0, hi0 = wilson(rules_right, n)
        lo1, hi1 = wilson(right, n)
        print(f"{name}: 95% interval    rules {100 * lo0:.0f}-{100 * hi0:.0f}%   rules + model {100 * lo1:.0f}-{100 * hi1:.0f}%")
        print(f"{name}: model fixed {fixed} lines and broke {broke} (net {fixed - broke:+d} of {n})")
    if not a.reserve:
        print("confusions:", ", ".join(f"{w}->{g} x{c}" for (w, g), c in conf.most_common(8)))


if __name__ == "__main__":
    main()
