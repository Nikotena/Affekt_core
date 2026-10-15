#!/usr/bin/env python3
"""
Fast review of many player lines (the 200-500 you write).

    python test/review_lines.py text.txt --out lines_review.csv
        -> runs the parser on every line and writes a CSV. Open it in Excel/Sheets and, ONLY
           for rows the parser got wrong, fill fix_intent (and fix_valence if the feeling is
           also wrong: -1, 0 or 1). Leave correct rows blank.

    python test/review_lines.py --score lines_review.csv [--save-labels labels.json]
        -> accuracy, the errors grouped by the rule that fired, and (optionally) a labelled
           set you can use as a regression test.

Numbering like "12. " at the start of a line is removed; blank lines and duplicates are skipped.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.IntentParser import INTENT_DELTAS, INTENTS, parse  # noqa: E402

COLUMNS = ["id", "text", "intent", "valence", "rule", "fix_intent", "fix_valence"]


def read_lines(path):
    seen, lines = set(), []
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            text = re.sub(r"^\s*\d+\s*[.)]\s*", "", raw.strip())
            if text and text.lower() not in seen:
                seen.add(text.lower())
                lines.append(text)
    return lines


def valence_of(intent):
    x = INTENT_DELTAS[intent][0]
    return 1 if x >= 0.1 else -1 if x <= -0.1 else 0


def write_review(lines, out_path):
    rows = []
    for i, text in enumerate(lines, 1):
        p = parse(text)
        rows.append([i, text, p.intent, p.valence, p.rule[:60], "", ""])
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    return rows


def labelled(path):
    """(text, intent, valence, predicted_intent, predicted_valence, rule) per row, with fixes applied."""
    out, problems = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for n, row in enumerate(csv.DictReader(f), 2):
            intent = (row.get("fix_intent") or "").strip() or row["intent"]
            if intent not in INTENTS:
                problems.append(f"row {n}: unknown intent {intent!r} (valid: {', '.join(INTENTS)})")
                continue
            fv = (row.get("fix_valence") or "").strip()
            if fv and fv not in ("-1", "0", "1", "+1"):
                problems.append(f"row {n}: fix_valence must be -1, 0 or 1, got {fv!r}")
                continue
            if fv:
                valence = int(fv)
            elif (row.get("fix_intent") or "").strip():
                valence = valence_of(intent)
            else:
                valence = int(row["valence"])
            out.append((row["text"], intent, valence, row["intent"], int(row["valence"]), row["rule"]))
    return out, problems


def score(path, save_labels=None):
    rows, problems = labelled(path)
    n = len(rows)
    if n == 0:
        for p in problems:
            print("WARNING:", p)
        print("no valid rows to score")
        return {"n": 0, "intent_acc": 0, "feeling_acc": 0, "opposite": 0, "problems": problems}
    ok_i = sum(r[1] == r[3] for r in rows)
    ok_v = sum(r[2] == r[4] for r in rows)
    opposite = sum(r[2] * r[4] < 0 for r in rows)
    other_rate = sum(r[3] == "other" for r in rows) / n if n else 0
    print(f"{n} lines reviewed")
    print(f"intent right:    {ok_i}/{n} ({ok_i / n:.0%})")
    print(f"feeling right:   {ok_v}/{n} ({ok_v / n:.0%})   opposite feeling: {opposite}")
    print(f"parser said 'other': {other_rate:.0%} of lines")
    wrong = Counter(r[5] for r in rows if r[1] != r[3] or r[2] != r[4])
    if wrong:
        print("\nrules behind the mistakes (most frequent first):")
        for rule, c in wrong.most_common(8):
            print(f"  {c:3d}  {rule}")
    confusion = Counter((r[1], r[3]) for r in rows if r[1] != r[3])
    if confusion:
        print("\nconfusions (expected -> parser):")
        for (a, b), c in confusion.most_common(10):
            print(f"  {c:3d}  {a} -> {b}")
    for p in problems:
        print("WARNING:", p)
    if save_labels:
        data = [{"text": r[0], "intent": r[1], "valence": r[2]} for r in rows]
        with open(save_labels, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"\nsaved {len(data)} labelled lines to {save_labels}")
    return {"n": n, "intent_acc": ok_i / n if n else 0, "feeling_acc": ok_v / n if n else 0,
            "opposite": opposite, "problems": problems}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input", nargs="?", help="text file with one player line per row")
    ap.add_argument("--out", default="lines_review.csv")
    ap.add_argument("--score", metavar="CSV", help="score a reviewed CSV instead of creating one")
    ap.add_argument("--save-labels", metavar="JSON")
    args = ap.parse_args()

    if args.score:
        score(args.score, args.save_labels)
        return
    if not args.input:
        ap.error("give a text file, or --score reviewed.csv")
    lines = read_lines(args.input)
    rows = write_review(lines, args.out)
    dist = Counter(r[2] for r in rows)
    print(f"{len(rows)} lines -> {args.out}")
    print("parser output:", ", ".join(f"{k} {v}" for k, v in dist.most_common()))
    print(f"'other' rate: {dist['other'] / len(rows):.0%}  (lines the rules did not understand)")


if __name__ == "__main__":
    main()
