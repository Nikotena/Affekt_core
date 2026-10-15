"""
Runtime fact guard: the model may word a price, but it may not change it.

The engine knows what each NPC sells ("You sell a Healing potion for 25 coins."). After the model
answers, `guard_price` looks at every amount of coins in the reply and compares it with the
engine's facts:

    ok         every amount is a price the NPC sells, or a number the player just said (a counter-offer)
               or the reply mentions no coins at all: the reply is kept
    corrected  the reply states a wrong price for something the NPC sells: replaced by a plain line
               with the engine's price ("The Healing potion costs 25 coins.")
    blocked    the reply names a price but the NPC has nothing to sell: replaced by `fallback`

Only amounts followed by a coin word are checked, so "three guards" or "at night, 10 minutes" are
left alone. Spelled-out amounts ("twenty-five coins") are converted to digits first.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from modules.TextNumbers import normalize_prices

_SELLS_RE = re.compile(r"You sell (?:an? |the |some )?(.+?) for (\d+) coins?\b", re.IGNORECASE)
_COINS_RE = re.compile(r"\b(\d+)\s*(?:gold\s+|silver\s+|copper\s+)?(?:coins?|gold|gp)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+\b")


@dataclass
class GuardResult:
    status: str          # "ok", "corrected" or "blocked"
    text: str            # the line to show the player
    stated: Tuple[int, ...] = ()
    expected: Tuple[int, ...] = ()


def known_prices(payload: List[str]) -> List[Tuple[str, int]]:
    """(item, price) pairs the engine put in the payload as facts."""
    found = []
    for line in payload:
        if line.startswith(("Last reply: ", "Player earlier: ")):  # earlier text, not a fact
            continue
        m = _SELLS_RE.search(line)
        if m:
            found.append((m.group(1).strip(), int(m.group(2))))
    return found


def stated_prices(reply: str) -> List[int]:
    """Amounts of coins the reply states, with spelled-out numbers converted to digits."""
    return [int(n) for n in _COINS_RE.findall(normalize_prices(reply))]


def guard_price(reply: str, payload: List[str], action: str, fallback: str) -> GuardResult:
    stated = stated_prices(reply)
    prices = known_prices(payload)
    expected = tuple(p for _, p in prices)
    if not stated:
        return GuardResult("ok", reply, (), expected)

    offered = {int(n) for n in _NUMBER_RE.findall(action)}  # the player's own numbers are fair to repeat
    allowed = set(expected) | offered
    if all(n in allowed for n in stated):
        return GuardResult("ok", reply, tuple(stated), expected)

    if prices:
        # the item the player named, else the first one the NPC sells
        item, price = next(((i, p) for i, p in prices if i.lower() in action.lower()), prices[0])
        return GuardResult("corrected", f"The {item} costs {price} coins.", tuple(stated), expected)
    return GuardResult("blocked", fallback, tuple(stated), expected)
