#!/usr/bin/env python3
"""Run the runtime fact guard over saved replies (rows with payload, action and reply).

    python test/check_guard.py data/raw_amd_big.jsonl [more.jsonl ...]

Prints, per file, how many replies state a price and what the guard did, then each reply it
changed. Zero changes on replies from a good model means no false alarms; a change shows a price
the model got wrong.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.FactGuard import guard_price, stated_prices  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    files = sys.argv[1:]
    if not files:
        sys.exit(__doc__)
    for path in files:
        counts, changed = Counter(), []
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                result = guard_price(row["reply"], row["payload"], row["action"], "<fallback>")
                counts["replies"] += 1
                counts["state a price"] += bool(stated_prices(row["reply"]))
                counts[result.status] += 1
                if result.status != "ok":
                    changed.append((row["action"], row["reply"], result.expected, result.stated, result.status))
        print(f"{path}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        for action, reply, expected, stated, status in changed:
            print(f"  {status}: engine {list(expected)} vs reply {list(stated)}: {reply}")


if __name__ == "__main__":
    main()
