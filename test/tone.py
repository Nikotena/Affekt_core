"""
Heuristic tone scoring for NPC lines (used by eval_llm.py).

Word-boundary regexes grouped by how a line sounds, so "fool" or "troublemaker"
count as hostile and "our" does not match inside "tour". Scores are hit counts; a
line passes its expected tone when that tone's hits beat the opposite tone's hits.
Generic words that both moods use ("don't", "fine", "best", "face") are left out.
"""

import re

_HOSTILE = [
    # insults
    "fool", "foolish", "idiot", "scum", "worm", "thief", "thieves", "beggar", "brute", "coward",
    "wretch", "pest", "vermin", "troublemaker", "trouble", "cretin", "imbecile", "your kind",
    "insolence", "greedy", "greed", "worthless", "pathetic", "nonsense", "mediocre", "insult", "disrespect",
    # threats
    "crush", "kill", "smash", "smashed", "break your", "regret", "carve", "melt", "scrap",
    "hang", "stocks", "dungeon", "banish", "banished", "teach you", "lose my temper",
    "lose my patience", "lose my mind", "make you pay", "pay for", "your bones", "or else",
    "last warning", "warn you", "i warn", "turn you into", "you'll be the one who dies",
    # commands to leave / back off
    "get out", "out of my", "leave", "go away", "move along", "step aside", "stand aside",
    "back off", "back away", "watch yourself", "watch your", "keep your distance",
    "keep away", "stay away", "scram", "begone", "get lost", "beat it", "walk on", "shut up",
    "not welcome", "keep your voice down", "keep your hands",
    # challenge / disdain
    "you dare", "how dare", "dare you", "think you can", "think you're", "think you are",
    "spare me", "i don't serve", "not interested", "charity", "waste of",
]

_FRIENDLY = [
    "welcome", "friend", "friends", "glad", "happy", "pleasure", "delighted", "nice", "great",
    "wonderful", "lovely", "good to see", "good day", "good morning", "good evening",
    "of course", "certainly", "gladly", "please", "thank", "thanks", "grateful", "appreciate",
    "kind", "dear", "come in", "come on in", "sit down", "take a seat", "on the house",
    "on me", "my treat", "at your service", "how can i help", "can i help", "let me help",
    "here to help", "help you", "what can i", "what'll it be", "anything else", "enjoy",
    "cheers", "back again", "good to have", "honored", "bless", "hello", "greetings",
    "delicious", "fresh",
]


def _compile(words):
    return [re.compile(r"(?<![\w'])" + re.escape(w) + (r"(?![\w])" if w[-1].isalnum() else ""),
                       re.IGNORECASE) for w in words]


_HOSTILE_RE = _compile(_HOSTILE)
_FRIENDLY_RE = _compile(_FRIENDLY)


def tone_scores(text: str) -> dict:
    """Hit counts per tone for a line."""
    text = text.replace("’", "'")
    return {
        "hostile": sum(1 for p in _HOSTILE_RE if p.search(text)),
        "friendly": sum(1 for p in _FRIENDLY_RE if p.search(text)),
    }


def matches_tone(text: str, mood: str) -> bool:
    """Whether the line sounds like the expected mood. Neutral/tense are not scored (True)."""
    s = tone_scores(text)
    if mood == "hostile":
        return s["hostile"] > 0 and s["hostile"] >= s["friendly"]
    if mood == "friendly":
        return s["friendly"] > 0 and s["friendly"] > s["hostile"]
    return True
