"""
Spelled-out numbers in NPC text.

Models like to write "twenty-five coins". For the player that is fine, but code that checks
prices needs digits, so `normalize_prices` rewrites them ("25 coins"). Shared by the dataset
filter and the evaluation benchmark.
"""

import re
from typing import Optional

_NUM_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
              "sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
              "hundred")
SPELLED_PRICE_RE = re.compile(rf"\b(?:(?:{_NUM_WORDS})[- ]?)+\s+(?:gold\s+)?coins?\b", re.IGNORECASE)

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_PRICE_PHRASE_RE = re.compile(
    rf"\b((?:{_NUM_WORDS})(?:(?:[- ]|[- ]and )(?:{_NUM_WORDS}))*)\s+((?:gold\s+)?coins?)\b", re.IGNORECASE)


def words_to_int(phrase: str) -> Optional[int]:
    """'seventy-five' -> 75, 'one hundred and twenty' -> 120. None if it is not a plain number."""
    total = current = 0
    for w in re.split(r"[-\s]+", phrase.lower().strip()):
        if w == "and":
            continue
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        else:
            return None
    return total + current if total + current > 0 else None


def normalize_prices(reply: str) -> str:
    """Rewrite spelled-out prices before a coin word as digits ("twelve coins" -> "12 coins")."""
    def fix(m):
        n = words_to_int(m.group(1))
        return m.group(0) if n is None else f"{n} {m.group(2)}"
    return _PRICE_PHRASE_RE.sub(fix, reply)
