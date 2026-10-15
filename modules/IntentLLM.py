"""
LLM fallback for the intent parser.

The rule based IntentParser is exact on clear lines but answers "other" for most chatty or
indirect ones (grumbling, small talk, gossip). `refine` asks a language model to pick one label
only for those lines, so the deterministic parser still decides everything it understands.
The model never changes engine state: it returns one label from INTENTS or nothing.
"""

import re
from dataclasses import replace
from typing import Optional

from modules.IntentParser import INTENT_DELTAS, INTENTS, ParsedAction

DEFINITIONS = {
    "greet": "says hello or calls out a friendly welcome",
    "farewell": "says goodbye or announces leaving",
    "ask": "asks a question or asks for information, news or directions",
    "ask_help": "asks the NPC for help, protection or rescue",
    "offer_help": "offers help, a service, food/drink or an opportunity to the NPC",
    "buy": "wants to buy or order something from the NPC",
    "haggle": "argues about the price or offers a different price",
    "sell": "wants to sell something to the NPC",
    "compliment": "praises, thanks or agrees warmly",
    "insult": "insults or belittles the NPC",
    "complain": "grumbles, vents or protests about something (prices, service, bad luck, the world, "
                "other people), including swearing and frustration, even without insulting the NPC",
    "threaten": "threatens the NPC with harm or consequences",
    "attack": "physically attacks or announces an attack on the NPC",
    "other": "none of the above: small talk, a statement, a story or gossip with no request or feeling aimed at the NPC",
}

SYSTEM_PROMPT = (
    "You classify what a player says to a video game NPC. Answer with exactly one label and nothing else.\n"
    "Labels:\n" + "\n".join(f"- {name}: {text}" for name, text in DEFINITIONS.items())
)

# longest names first so "ask_help" is not read as "ask"
_LABEL_RE = re.compile(r"\b(" + "|".join(sorted(INTENTS, key=len, reverse=True)) + r")\b")


def parse_label(text: str) -> Optional[str]:
    m = _LABEL_RE.search((text or "").lower())
    return m.group(1) if m else None


def classify(backend, text: str) -> Optional[str]:
    """One label for `text`, or None if the model failed or answered something unusable."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    try:
        return parse_label(backend.complete(messages, max_tokens=12, temperature=0.0))
    except Exception:
        return None


def refine(parsed: ParsedAction, backend) -> ParsedAction:
    """Keep the parser's answer unless it was "other"; then let the model pick a label."""
    if parsed.intent != "other":
        return parsed
    label = classify(backend, parsed.clean or parsed.text)
    if not label or label == "other":
        return parsed
    return replace(parsed, intent=label, delta=list(INTENT_DELTAS[label]), rule="llm")
