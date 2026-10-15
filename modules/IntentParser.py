"""
Intent Parser (deterministic, rule based)

Turns what the player typed into an intent, the emotional push it gives the NPC, and the
numbers/items it mentions. First match in PRIORITY order wins, so a threat hidden behind a
compliment ("Nice shop. Shame if something happened to it.") is read as a threat, and
"Hello, I'd like to buy a potion" is a purchase, not just a greeting.

Pure standard library, no model, same answer every time.
"""

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

INTENTS = ["greet", "farewell", "ask", "ask_help", "offer_help", "buy", "haggle", "sell",
           "compliment", "insult", "complain", "threaten", "attack", "other"]

# Emotional push per intent: [valence X, initiative Y, stress Z, inertia W].
# greet / attack / offer_help keep the engine's original values.
INTENT_DELTAS: Dict[str, List[float]] = {
    "greet": [0.3, -0.2, 0.1, 0.7],
    "farewell": [0.15, -0.1, 0.0, 0.5],
    "ask": [0.05, 0.05, 0.0, 0.5],
    "ask_help": [0.05, -0.1, 0.2, 0.5],
    "offer_help": [0.4, -0.1, 0.2, 0.6],
    "buy": [0.05, 0.0, 0.0, 0.5],
    "haggle": [-0.05, 0.1, 0.1, 0.5],
    "sell": [0.05, 0.0, 0.0, 0.5],
    "compliment": [0.35, -0.1, -0.05, 0.6],
    "insult": [-0.35, 0.2, 0.4, 0.8],
    "complain": [-0.2, 0.1, 0.3, 0.6],
    "threaten": [-0.4, 0.3, 0.7, 0.9],
    "attack": [-0.4, 0.3, 0.8, 0.9],
    "other": [0.0, 0.0, 0.0, 0.5],
}

COMMERCE_INTENTS = {"buy", "haggle", "sell"}
COMMERCE_WORDS = ("sell", "buy", "trade", "purchase", "price", "cost", "how much")


def _rx(*patterns: str):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# Phrases that cancel a violent verb: "I won't fight you", "I'm not here to hurt anyone".
_NEGATED_VIOLENCE = re.compile(
    r"\b(?:i (?:won't|will not|wouldn't|would never|never|don't|do not|can't|cannot|am not going to|'m not going to)|"
    r"(?:i'm|i am) not (?:here|going|about) to|not here to|no intention of|mean no)\b(?:\s+\w+){0,3}?\s+"
    r"(?:fight|attack|hurt|harm|kill|hit|strike|rob|steal|use)\b(?:\s+it)?(?:\s+on\s+\w+)?", re.IGNORECASE)

# Chat slang the rules would otherwise miss
_SLANG = [(re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE), v) for k, v in {
    "gimme": "give me", "wanna": "want to", "gonna": "going to", "gotta": "got to", "lemme": "let me",
    "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please", "thx": "thanks",
    "ya": "you", "y'all": "you", "cuz": "because"}.items()]

_PLACEHOLDER = re.compile(r"\[[^\]]*\]")
_PLACEHOLDER_SRC = r"\[[^\]]*\]"  # "[NPC_Name]": the player addresses the NPC by name


def strip_vocatives(text: str, names: Iterable[str] = ()):
    """Remove the NPC's name/nickname when the player is just addressing it.

    "Hey Marta, how much for the potion?" -> ("Hey, how much for the potion?", "Marta").
    Only vocative positions are touched (start of the line, or set off by commas), so a name
    that is also a noun ("the Guard") keeps its meaning inside a sentence.
    Returns (clean_text, name_used_or_None).
    """
    alternatives = [re.escape(n) for n in sorted({n for n in names if n}, key=len, reverse=True)]
    alt = "|".join(alternatives + [_PLACEHOLDER_SRC])
    addressed = None

    lead = re.compile(
        rf"^\s*((?:hey|hi|hello|oh|ah|yo|excuse me|listen|look)\b[,]?\s+)?({alt})(?![\w])[,!:]?\s*", re.IGNORECASE)
    m = lead.match(text)
    if m and (m.group(1) or text[m.end():].strip() or True):
        addressed = m.group(2)
        greeting = (m.group(1) or "").strip().rstrip(",")
        text = (greeting + ", " if greeting else "") + text[m.end():]

    inner = re.compile(rf"\s*,\s*({alt})(?![\w])\s*(?=[,.!?]|$)", re.IGNORECASE)
    m2 = inner.search(text)
    if m2:
        addressed = addressed or m2.group(1)
        text = inner.sub("", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text, (None if addressed and addressed.startswith("[") else addressed)
_OPENER = re.compile(
    r"^(?:(?:excuse me|pardon me|pardon|sorry|listen|look|wait|hold on|um+|uh+|okay|ok)\b[,.!]?\s*|well,\s*)+",
    re.IGNORECASE)

_PEACE = _rx(r"\bcome in peace\b", r"\bin peace\b", r"\bmean no harm\b", r"\bno harm\b")

# (intent, patterns) in priority order
RULES = [
    ("attack", _rx(
        r"\bi (?:\w+ ){0,3}?(?:stab|punch|kick|slash|shoot|slap|smash|shove|tackle|club|bash|whack|impale|behead)\b",
        r"\bi (?:draw|unsheathe|swing|raise|brandish) (?:my |the )?(?:sword|axe|blade|dagger|weapon|fist|mace|bow|knife)",
        r"\bi (?:strike|hit|attack|throw|hurl|lunge|charge)\b(?: \w+){0,5}? (?:at|you|him|her|them|the \w+)",
        r"\bswing (?:my )?\w+ at\b",
        r"^(?:attack|fight|hit|strike|kill|stab|punch|kick)\b",
        r"\btaste my (?:blade|steel|fist|wrath|sword)\b", r"^die\b", r"\bprepare to die\b",
        r"\ben garde\b", r"\bhave at you\b", r"\bfight me\b", r"\blet'?s fight\b", r"\bi challenge you\b",
        r"\battack\W*$", r"\bget (?:him|her|them)\b",
        r"\bi (?:cast|hurl|fire|launch|loose) (?:\w+ ){0,3}at\b",
    )),
    ("threaten", _rx(
        r"\bor (?:else|i(?:'ll| will)|you(?:'ll| will))\b",
        r"\bi(?:'ll| will| am going to|'m going to) (?:kill|burn|hurt|make you|end you|destroy|ruin|find you|break|crush|"
        r"hunt|come for|bury|skin|gut|cut)\b",
        r"\byou(?:'d| had| would)? better (?:watch|not|run|be careful|keep|stay|leave|start|pray)\b",
        r"\bwatch your back\b", r"\bwatch yourself\b",
        r"\b(?:shame|pity) if\b", r"\bwould be a (?:shame|pity)\b", r"\bwouldn't want .{0,30}to (?:happen|burn|break)\b",
        r"\bknow where (?:you|your)\b", r"\bwon't like what\b", r"\bdon't make me\b", r"\blast warning\b",
        r"\byou(?:'ll| will) (?:regret|pay|suffer|lose|be sorry|die)\b", r"\bi(?:'ll| will) find you\b",
        r"\bmake you regret\b", r"\bor you(?:'ll| will)\b",
        r"\bgive me (?:your|all your|everything you)\b", r"\bhand (?:it |them )?over\b",
        r"\bempty your (?:pockets|purse)\b", r"\bstand and deliver\b", r"\byour (?:money|gold|purse) or\b",
    )),
    ("insult", _rx(
        r"\b(?:stupid|idiotic|pathetic|worthless|useless|incompetent|disgusting|garbage|rubbish|good-for-nothing|"
        r"dumbass|jackass|asshole|dickhead|idiot|moron|imbecile|numbskull|dimwit)\b",
        r"\byou(?:'re| are)? .{0,30}\b(?:idiot|fool|coward|peasant|scum|moron|dolt|filth|swine|wretch|ugly|liar|"
        r"lazy|dumb|imbecile|buffoon|clown|beggar|brute|crook|cheat|fraud|swindler|disgrace)\b",
        r"\b(?:filthy|lazy|dirty|stinking|miserable) (?:beggar|peasant|fool|dog|rat|swine|thief|liar)\b",
        r"\byou call (?:this|that)\b", r"\btastes? like (?:pig|dirt|mud|garbage|swill|water)\b", r"\bbetter manners\b",
        r"\bno manners\b", r"\bis that the best you\b", r"\bout of my (?:way|sight)\b", r"\bget lost\b",
        r"\bshut (?:up|your mouth)\b", r"\bwhat a (?:joke|waste|disgrace|useless|pathetic)\b",
        r"\bpeasant\b", r"\bold man\b.*\bbest\b",
        r"\bgo to hell\b", r"\bdrop dead\b", r"\brot in\b", r"\bhow dare you\b", r"\byou (?:smell|stink|suck)\b",
        r"\bi(?:'d| would) rather (?:die|starve|rot)\b", r"\bthe best you (?:can|could|have|got|'ve)\b",
        r"\bis that all you\b",
    )),
    ("complain", _rx(
        r"\bdamn it\b", r"\bdammit\b", r"\bfor (?:god's|goodness|heaven's|pete's) sake\b", r"\bfor crying out loud\b",
        r"\b(?:what|who|why|where|how) the (?:hell|heck|fuck)\b", r"\bare you (?:kidding|serious|deaf|blind|joking)\b",
        r"\bfed up\b", r"\bsick of\b", r"\bso now\b", r"\bor what\?", r"\bwhat'?s (?:wrong with you|your problem)\b",
        r"\bi'?ve been waiting\b", r"\bis anyone (?:actually |even )?(?:working|there|home|listening)\b",
        r"\bunbelievable\b", r"\bi can'?t believe (?:you|this)\b", r"\bseriously\?",
    )),
    ("ask_help", _rx(
        r"\b(?:help|save|rescue|protect|hide|shelter|guide) me\b", r"\bi need (?:your )?(?:help|a healer|a doctor|a medic|"
        r"shelter|a guide|assistance|rescue|protection)\b",
        r"\b(?:can|could|would|will) you (?:please )?(?:help|hide|save|shelter|protect|rescue|heal)\b",
        r"\bi(?:'m| am) (?:lost|hurt|wounded|injured|in trouble|in danger)\b",
        r"\b(?:wounded|injured|bleeding|dying|poisoned)\b", r"\bplease,? help\b",
        r"\bdon'?t hurt me\b", r"\bspare me\b", r"\bmercy\b", r"\bplease,? don'?t\b",
    )),
    ("offer_help", _rx(
        r"\bneed a hand\b", r"\bwant me to\b", r"\blet me (?:help|carry|fix|take|handle|do|get)\b", r"\ballow me\b",
        r"\bhow can i (?:be of service|help)\b", r"\b(?:can|may) i help\b", r"\bat your service\b",
        r"\bi(?:'ll| will| can| could) (?:gladly |happily |easily )?(?:help|fix|repair|carry|handle|deal with|"
        r"take care of|assist|mend|clear|escort|guard)\b",
        r"\bi(?:'d| would) be (?:glad|happy|willing|honored) to\b", r"^(?:help|aid|assist)\b",
        r"\boffer (?:my )?(?:aid|help|assistance)\b",
    )),
    ("haggle", _rx(
        r"\btoo (?:expensive|pricey|high)\b", r"\btoo much (?:for|money|gold|coin)", r"\brobbery\b", r"\brip.?off\b", r"\boverpriced\b",
        r"\blower (?:the|your) price\b", r"\bcheaper\b", r"\bdiscount\b", r"\bknock .{0,15}off\b", r"\bbargain\b",
        r"\bhaggle\b", r"\bnegotiate\b", r"\bdo better\b", r"\bcan(?:'t|not) (?:afford|pay)\b", r"\bat most\b",
        r"\bhow about \d+", r"\bi(?:'ll| will) (?:give|pay|offer) (?:you )?\d+",
    )),
    ("sell", _rx(
        r"\bto sell\b", r"\bi(?:'ll| will) sell\b", r"\bsell (?!me\b)(?:my|this|these|that|some)\b", r"\bfor sale\b",
        r"\bwould you (?:buy|pay|take)\b", r"\bdo you (?:buy|want to buy)\b", r"\binterested\?",
        r"\binterested in (?:buying|these|this)\b", r"\blooking to (?:sell|trade)\b", r"\btrade my\b",
        r"\bbuy (?:this|these|my) .{0,30}from me\b", r"^sell\b", r"^trade\b", r"\btrade with\b",
    )),
    ("buy", _rx(
        r"\bbuy\b", r"\bpurchase\b", r"\bi(?:'ll| will) take\b", r"\bsell me\b", r"\bgot any\b",
        r"\bgive me (?:a|an|the|some|two|three|four|five|\d+)\b",
        r"\bi (?:want|need|would like|wish) (?:a|an|the|some|two|three|four|five|\d+)\b",
        r"\bi(?:'d| would) like (?:a|an|the|some|two|three|four|five|\d+)\b", r"\bi need some\b",
        r"\bdo you (?:have|sell|stock|carry)\b(?!.{0,15}\b(?:news|work|rumou?rs?|information|quests?|jobs?)\b)",
        r"^(?:a|an|one|two|three|four|five|\d+)\b[\w ]{0,30},? please\b",
    )),
    ("farewell", _rx(
        r"\b(?:goodbye|good-bye|farewell|bye|see you|take care|stay safe|until next time|so long|good night|safe travels)\b",
        r"\bi (?:should|must|have to|need to|better|got to|gotta) (?:go|leave|be (?:going|off|leaving)|get going|head out|head off)\b",
        r"\btime to (?:go|leave|head)\b", r"\bi(?:'m| am) (?:leaving|off|going now|heading out)\b",
        r"\bi(?:'ll| will) be (?:going|off|leaving)\b",
    )),
    ("compliment", _rx(
        r"\b(?:thank you|thanks|grateful|appreciate|well done|admire|bless you|you saved|kindness|good job|great job)\b",
        r"\byou(?:'re| are) (?:amazing|wonderful|great|the best|brave|kind|generous)\b",
    )),
    ("greet", _rx(
        r"^(?:hi|hello|hey|yo|greetings|howdy|salutations|hail|good (?:morning|evening|day|afternoon)|morning|"
        r"evening|afternoon|well met)\b",
        r"\b(?:hello|greetings|howdy)\b", r"^greet\b", r"\bgreet the\b",
        r"\bhow are you\b", r"\bhow's it going\b", r"\bwhat's up\b", r"\bhow do you do\b",
    )),
    ("ask", _rx(
        r"^(?!what an?\b)(?:where|what|who|when|why|how|which|whose|whom)\b",
        r"\b(?:tell me|do you know|did you (?:see|hear)|have you (?:seen|heard)|know anything|any (?:news|rumou?rs?)|"
        r"can you tell|could you tell|i(?:'m| am) (?:looking for|searching for|trying to find)|i wonder)\b",
        r"^(?:ask|inquire)\b", r"\bsince when\b", r"\bhow come\b", r"\bwhat happened\b",
        r"^(?:do|did|does|can|could|will|would|is|are|was|were|have|has|may|should) (?:you|they|he|she|there|it|anyone)\b",
    )),
]

# Praise needs an adjective *and* a "you/your": "Nice sword, did you forge it?" but not "fine, I'll go".
_PRAISE_ADJ = re.compile(
    r"\b(?:fine|great|excellent|wonderful|lovely|best|amazing|impressive|beautiful|delicious|superb|magnificent|"
    r"marvelous|brilliant|skilled|talented|nice|love|adore|kindest|kind|generous|noble|wise|brave)\b", re.IGNORECASE)
_SARCASTIC_BEST = re.compile(r"\bthe best you (?:can|could|have|got|'ve)\b", re.IGNORECASE)
_YOU =re.compile(r"\byou(?:r|'re|rs)?\b", re.IGNORECASE)
_OFFER = re.compile(r"(\d+)\s*(?:gold\s+)?coins?\b|\bfor (\d+)\b|\bhow about (\d+)\b|\bgive you (\d+)\b", re.IGNORECASE)


def _first_match(name: str, patterns, text: str) -> Optional[str]:
    for p in patterns:
        if p.search(text):
            return p.pattern
    praise_text = _SARCASTIC_BEST.sub(" ", text)
    if name == "compliment" and _PRAISE_ADJ.search(praise_text) and _YOU.search(praise_text):
        return "praise+you"
    return None


@dataclass
class ParsedAction:
    text: str
    intent: str
    delta: List[float]
    rule: str
    offer: Optional[int] = None
    item: Optional[str] = None
    clean: str = ""                     # text without the name/nickname the player addressed the NPC with
    addressed: Optional[str] = None     # that name, if the player used one

    @property
    def valence(self) -> int:
        """-1 / 0 / +1: whether the NPC should feel the line as hostile, neutral or pleasant."""
        x = self.delta[0]
        return 1 if x >= 0.1 else -1 if x <= -0.1 else 0


class IntentParser:
    def __init__(self, items: Iterable[str] = ()):
        self.items = [i for i in items if i]

    def parse(self, text: str, names: Iterable[str] = ()) -> ParsedAction:
        raw = (text or "").strip()
        raw_clean, addressed = strip_vocatives(raw, names)
        low = raw_clean.lower().replace("’", "'")
        for pattern, repl in _SLANG:
            low = pattern.sub(repl, low)
        low = _PLACEHOLDER.sub(" ", low)
        low = re.sub(r"\s+", " ", low).strip()
        low = _OPENER.sub("", low)
        cleaned = _NEGATED_VIOLENCE.sub(" ", low)

        intent, rule = self._classify(cleaned, low)

        # "Hi! How much for the sword?": a greeting followed by something real is about the something
        if intent == "greet":
            rest = re.split(r"(?<=[,.!?])\s+", cleaned, maxsplit=1)
            if len(rest) == 2:
                sub_intent, sub_rule = self._classify(rest[1], rest[1])
                if sub_intent not in ("other", "greet"):
                    intent, rule = sub_intent, sub_rule

        offer = self._offer(cleaned)
        if intent == "buy" and offer is not None:
            intent, rule = "haggle", "buy+offer"

        return ParsedAction(raw, intent, list(INTENT_DELTAS[intent]), rule, offer, self._item(low),
                            clean=raw_clean, addressed=addressed)

    @staticmethod
    def _classify(cleaned: str, low: str):
        for name, patterns in RULES:
            hit = _first_match(name, patterns, cleaned)
            if hit:
                return name, hit
        if any(p.search(low) for p in _PEACE):
            return "greet", "peace"
        return "other", "default"

    @staticmethod
    def _offer(text: str) -> Optional[int]:
        m = _OFFER.search(text)
        return int(next(g for g in m.groups() if g)) if m else None

    def _item(self, low: str) -> Optional[str]:
        for item in self.items:
            name = item.lower()
            if re.search(rf"\b{re.escape(name)}s?\b", low):
                return item
        return None


def is_commerce(action: str, parsed: Optional[ParsedAction] = None) -> bool:
    """Whether the NPC should mention what it sells: shared by the engine and the dataset generator."""
    low = action.lower()
    if any(w in low for w in COMMERCE_WORDS):
        return True
    parsed = parsed or _DEFAULT.parse(action)
    return parsed.intent in COMMERCE_INTENTS


_DEFAULT = IntentParser()


def parse(text: str, names: Iterable[str] = ()) -> ParsedAction:
    return _DEFAULT.parse(text, names)
