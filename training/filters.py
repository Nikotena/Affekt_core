"""
Quality filter for teacher replies. Reuses the eval benchmark's checks and adds the
rules that matter for training data: no stage directions, no third person, no name
prefixes, no reasoning leaks, no reciting the NPC's own prompt facts, no profanity
(unless allowed), minimum length, and near-duplicate removal.
"""

import os
import re
import sys
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test"))

from eval_llm import run_checks  # noqa: E402

STAGE_RE = re.compile(r"[*\[\]()<>_~{}]")
THIRD_PERSON_RE = re.compile(
    r"^\W*(he|she|they|the (guard|merchant|blacksmith|healer|bartender|npc|man|woman))\b|"
    r"\b(he|she) (says|growls|grunts|snarls|smiles|nods|sighs)\b", re.IGNORECASE)
PREFIX_RE = re.compile(r"^\s*[A-Z][\w ]{0,20}:\s")
REASONING_RE = re.compile(
    r"\b(i should|i need to|i must (respond|reply|answer)|the user|"
    r"the player (wants|is|does|asks|offers|greets|attacks)|as an? (npc|ai)|my (role|task|goal) is)\b",
    re.IGNORECASE)
PROFANITY_RE = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|cunt|asshole|dickhead|motherfucker)\b", re.IGNORECASE)
MIN_WORDS = 3


def echoes_identity(reply: str, name: Optional[str], role: Optional[str]) -> bool:
    """True if the NPC recites its own prompt facts ("I'm Lorcan, the Miller")."""
    if not name:
        return False
    if re.search(rf"\b(i'?m|i am|my name is|name's)\s+{re.escape(name)}\b", reply, re.IGNORECASE):
        return True
    return bool(role and re.search(rf"\b(i'?m|i am) (a|an|the) {re.escape(role)}\b", reply, re.IGNORECASE))


# Place names distinctive enough to catch "wrong place" (Gate/Farm/Square/Inn are common words)
PLACE_WORDS = ["tavern", "market", "forge", "chapel", "docks", "workshop"]
PRICE_RE = re.compile(r"\b(\d+)\s*(?:gold\s+)?coins?\b", re.IGNORECASE)


def consistency_problem(reply: str, place: Optional[str], price: Optional[int],
                        price_shown: bool) -> Optional[str]:
    """Facts the reply invents or contradicts: another place, or a price the prompt didn't give."""
    low = reply.lower()
    if place:
        for other in PLACE_WORDS:
            if other != place.lower() and re.search(rf"\b{other}\b", low):
                return "wrong_place"
    for amount in PRICE_RE.findall(reply):
        if not price_shown or (price is not None and int(amount) != price):
            return "invented_price"
    return None


def reject_reason(reply: str, mood: str, expected: List[str], name: Optional[str] = None,
                  role: Optional[str] = None, allow_profanity: bool = False,
                  place: Optional[str] = None, price: Optional[int] = None,
                  price_shown: bool = False) -> Optional[str]:
    """First reason the reply is unusable as training data, or None if it is clean."""
    if "�" in reply:
        return "bad_encoding"
    problem = consistency_problem(reply, place, price, price_shown)
    if problem:
        return problem
    if REASONING_RE.search(reply):
        return "reasoning_leak"
    if echoes_identity(reply, name, role):
        return "recites_identity"
    if not allow_profanity and PROFANITY_RE.search(reply):
        return "profanity"
    if STAGE_RE.search(reply):
        return "stage_direction"
    if THIRD_PERSON_RE.search(reply):
        return "third_person"
    if PREFIX_RE.match(reply):
        return "name_prefix"
    if len(reply.split()) < MIN_WORDS:
        return "too_short_words"
    failed = [name_ for name_, ok in run_checks(reply, mood, expected).items() if not ok]
    return failed[0] if failed else None


def _grams(text: str, n: int = 3):
    words = re.findall(r"[a-z']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)} or {tuple(words)}


def drop_near_duplicates(rows: Iterable[Dict], threshold: float = 0.6) -> List[Dict]:
    """Keep the first of any group of replies whose 3-gram Jaccard similarity >= threshold."""
    kept, kept_grams = [], []
    for row in rows:
        g = _grams(row["reply"])
        if any(len(g & k) / len(g | k) >= threshold for k in kept_grams):
            continue
        kept.append(row)
        kept_grams.append(g)
    return kept
