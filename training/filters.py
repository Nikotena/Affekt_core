"""
Quality filter for teacher replies. Reuses the eval benchmark's checks and adds the
rules that matter for training data: no stage directions, no third person, no name
prefixes, minimum length, and near-duplicate removal.
"""

import os
import re
import sys
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test"))

from eval_llm import run_checks  # noqa: E402

STAGE_RE = re.compile(r"[*\[\]()<>_~]|\{|\}")
THIRD_PERSON_RE = re.compile(
    r"^\W*(he|she|they|the (guard|merchant|blacksmith|healer|bartender|npc|man|woman))\b|"
    r"\b(he|she) (says|growls|grunts|snarls|smiles|nods|sighs)\b", re.IGNORECASE)
PREFIX_RE = re.compile(r"^\s*[A-Z][\w ]{0,20}:\s")
MIN_WORDS = 3


def reject_reason(reply: str, mood: str, expected: List[str]) -> Optional[str]:
    """First reason the reply is unusable as training data, or None if it is clean."""
    if "�" in reply:
        return "bad_encoding"
    if STAGE_RE.search(reply):
        return "stage_direction"
    if THIRD_PERSON_RE.search(reply):
        return "third_person"
    if PREFIX_RE.match(reply):
        return "name_prefix"
    if len(reply.split()) < MIN_WORDS:
        return "too_short_words"
    failed = [name for name, ok in run_checks(reply, mood, expected).items() if not ok]
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
