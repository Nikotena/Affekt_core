"""
Scenario grid for the fine-tuning dataset.

Each scenario describes one NPC moment (who, where, mood, what the player does, what is
for sale). `make_payload` rebuilds the exact payload lines the engine would produce, so
training prompts match what the model sees in game (train/serve parity).
"""

import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from modules.LLMLayer import style_for

# Game roles first (names differ from the game's so bench prompts never repeat verbatim).
ROLES = [
    "Bartender", "Merchant", "Guard", "Healer", "Blacksmith",
    "Baker", "Fisherman", "Hunter", "Clerk", "Beekeeper", "Tailor", "Innkeeper", "Scholar",
    "Farmer", "Sailor", "Carpenter", "Miller", "Cook", "Messenger", "Potter",
    # held out entirely from training (see HELD_OUT_ROLES)
    "Jeweler", "Cartographer", "Glassblower", "Priest",
]
HELD_OUT_ROLES = {"Jeweler", "Cartographer", "Glassblower", "Priest"}

NAMES = [
    "Oswin", "Brenna", "Tavish", "Lena", "Cormac", "Ilse", "Rurik", "Maeve", "Dorian", "Sela",
    "Garrick", "Nora", "Fenn", "Petra", "Ansel", "Mireille", "Joss", "Talia", "Edric", "Wren",
    "Bastian", "Odette", "Corin", "Yara", "Hale", "Isolde", "Pike", "Neve", "Lorcan", "Thea",
]

PLACES = ["Tavern", "Market", "Forge", "Gate", "Chapel", "Docks", "Farm", "Workshop", "Square", "Inn"]

# (item, fixed price or None). Game items keep their game prices.
ITEMS = [
    ("Potion", 25), ("Sword", 100),
    ("Loaf of bread", None), ("Candle", None), ("Dagger", None), ("Cloak", None), ("Boots", None),
    ("Shield", None), ("Bow", None), ("Bandage", None), ("Herb bundle", None), ("Map", None),
    ("Lamp", None), ("Waterskin", None), ("Hammer", None), ("Net", None), ("Cheese wheel", None),
    ("Ring", None), ("Scroll", None), ("Torch", None), ("Backpack", None), ("Apple basket", None),
    # held out entirely from training (see HELD_OUT_ITEMS)
    ("Compass", None), ("Amulet", None), ("Telescope", None), ("Saddle", None), ("Flute", None),
    ("Mirror", None),
]
HELD_OUT_ITEMS = {"Compass", "Amulet", "Telescope", "Saddle", "Flute", "Mirror"}

MOODS = ["friendly", "hostile", "neutral", "tense"]

# kind -> raw player actions (what the engine would receive)
ACTIONS: Dict[str, List[str]] = {
    "greet": ["greet the NPC", "say hello"],
    "attack": ["attack the NPC", "fight the NPC"],
    "help": ["help the NPC", "offer aid"],
    "buy": ["buy a {item} from me", "I want to buy the {item}", "purchase the {item}"],
    "trade": ["trade with the NPC", "sell something"],
    "ask": ["ask about the quest", "ask a question", "ask about the town"],
    "other": ["dance", "stare silently", "whistle a tune"],
}
# sampling weights per kind (buy matters most for the facts check)
KIND_WEIGHTS = {"greet": 3, "attack": 3, "help": 2, "buy": 4, "trade": 1, "ask": 3, "other": 1}

INITIATIVES = [-0.3, 0.0, 0.3]
KNOWS_SINCE = ["Act I", "Act II", None, None]


@dataclass
class Scenario:
    id: str
    name: str
    role: str
    place: str
    mood: str
    initiative: float
    action: str
    kind: str
    item: Optional[str] = None
    price: Optional[int] = None
    knows_since: Optional[str] = None
    knows_player: bool = False
    split: str = "train"
    expected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _commerce(action: str) -> bool:
    """Mirror of KankeiEngine._graph_facts: when the NPC mentions what it sells."""
    low = action.lower()
    return any(w in low for w in ("sell", "give", "unlock", "buy", "trade", "price", "ask", "purchase"))


def make_payload(sc: Scenario) -> List[str]:
    """Payload lines in the same order and wording as KankeiEngine.build_payload."""
    lines = ["World: Tavern", f"NPC: {sc.name}", f"You are {sc.name}, the {sc.role}.",
             f"You are at the {sc.place}."]
    if sc.knows_player:
        lines.append("You know the player" + (f" since {sc.knows_since}." if sc.knows_since else "."))
    if sc.item and _commerce(sc.action):
        lines.append(f"You sell a {sc.item} for {sc.price} coins.")
    lines.append(style_for(sc.mood, sc.initiative))
    return lines


def _split_for(role: str, item: Optional[str], rng: random.Random, val_rate: float) -> str:
    if role in HELD_OUT_ROLES or (item in HELD_OUT_ITEMS):
        return "eval_unseen"
    return "eval_seen" if rng.random() < val_rate else "train"


def generate_scenarios(n: int, seed: int = 0, val_rate: float = 0.08,
                       exclude_keys: Optional[set] = None) -> List[Scenario]:
    """Random scenarios. `exclude_keys` holds (role, mood, kind, item) tuples to skip
    (the eval benchmark's scenarios)."""
    rng = random.Random(seed)
    kinds, weights = list(KIND_WEIGHTS), list(KIND_WEIGHTS.values())
    out, seen = [], set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        role = rng.choice(ROLES)
        mood = rng.choice(MOODS)
        kind = rng.choices(kinds, weights)[0]
        item, fixed = rng.choice(ITEMS)
        price = fixed if fixed is not None else rng.choice([2, 3, 5, 7, 8, 12, 15, 18, 20, 30, 35, 45, 60, 75, 120, 150])
        raw = rng.choice(ACTIONS[kind]).format(item=item.lower())
        key = (role, mood, kind, item)
        if exclude_keys and (key in exclude_keys or key[:3] in exclude_keys):
            continue
        uid = (role, mood, raw, item, price)
        if uid in seen:
            continue
        seen.add(uid)
        knows = rng.random() < 0.4
        sc = Scenario(
            id=f"s{seed}_{len(out):05d}", name=rng.choice(NAMES), role=role,
            place=rng.choice(PLACES), mood=mood, initiative=rng.choice(INITIATIVES),
            action=raw, kind=kind, item=item, price=price,
            knows_player=knows, knows_since=rng.choice(KNOWS_SINCE) if knows else None,
            split=_split_for(role, item, rng, val_rate),
        )
        if kind in ("buy", "trade") and _commerce(raw):
            sc.expected = [str(price)]
        out.append(sc)
    return out
