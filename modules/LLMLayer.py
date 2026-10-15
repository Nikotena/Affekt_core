"""
LLM Layer (the "20" of the 80/20 split)

Turns the deterministic payload into natural language. The model never changes
engine state: it only phrases what the deterministic core already decided.
Target model: Qwen3-0.6B (GGUF, Q8_0) running locally through llama-cpp-python.
"""

import os
import random
import re
from typing import Dict, List, Optional

from modules.IntentParser import parse

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "model", "Qwen3-0.6B-Q8_0.gguf")

SYSTEM_PROMPT = (
    "You are the voice of a video game NPC. Reply in character with ONE or TWO short "
    "sentences, in the player's language. Use only the facts given. Never mention "
    "numbers, systems or that you are an AI."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Deterministic fallback lines (used when no LLM is needed or the LLM fails)
FALLBACK_LINES = {
    "friendly": "Welcome, friend. What can I do for you?",
    "hostile": "Watch your step. I have no patience for you.",
    "tense": "Keep your voice down... something feels wrong.",
    "neutral": "Hm. What do you need?",
}


# (keywords, second-person narration) - first match wins
_INTENT_NARRATION = {
    "attack": "The player attacks you.",
    "threaten": "The player threatens you.",
    "insult": "The player insults you.",
    "complain": "The player complains angrily at you.",
    "greet": "The player greets you.",
    "farewell": "The player says goodbye to you.",
    "offer_help": "The player offers to help you.",
    "ask_help": "The player asks you for help.",
    "buy": "The player wants to buy something from you.",
    "haggle": "The player tries to haggle over the price.",
    "sell": "The player wants to trade with you.",
    "ask": "The player asks you a question.",
    "compliment": "The player compliments you.",
}

# The deterministic core decides HOW the NPC sounds; the model only words it.
MOOD_STYLE = {
    "friendly": "warm and welcoming; offer something",
    "hostile": "openly hostile, rude and menacing; insult or threaten the player; never polite",
    "tense": "nervous and wary; short, guarded sentences",
    "neutral": "neutral and businesslike",
}

# Few-shot pool: (mood, facts, player action, NPC line). Each call shows a few examples of
# the current mood, one of them for the same kind of action. Rules for writing them:
#  - invent a different item and price in every example (never the game's real ones), so
#    prices and objects cannot be copied into a reply;
#  - vary openers and closers, models reuse what they see;
#  - keep neutral lines specific and two sentences long, short ones get copied verbatim.
_EXAMPLES = (
    ("friendly", ["You are Alan, the Bartender.", "You know the player."],
     "The player greets you.", "Look who's back! Sit down, I'll pour you something."),
    ("friendly", ["You are Dara, the Fisherman.", "You sell a Lantern for 14 coins."],
     "The player wants to buy something from you.", "Of course, friend! The lantern is 14 coins, and it burns all night."),
    ("friendly", ["You are Mira, the Healer."],
     "The player offers to help you.", "Bless you, traveler. A steady pair of hands is just what I need today."),
    ("friendly", ["You are Tomas, the Clerk."],
     "The player asks you a question.", "Happy to answer! The old road is safest at dawn."),
    ("friendly", ["You are Bran, the Guard.", "You know the player."],
     "The player greets you.", "Evening, citizen! Quiet night so far, thanks to folks like you."),
    ("friendly", ["You are Hilda, the Baker."],
     "The player attacks you.", "Oh my, please stop! Whatever this is, we can sort it out over bread."),
    ("hostile", ["You are Kael, the Blacksmith."],
     "The player attacks you.", "Bold of you, swinging at me. I have buried bigger fools than you."),
    ("hostile", ["You are Bran, the Guard."],
     "The player greets you.", "Keep walking. I dislike your face, and your chatter even more."),
    ("hostile", ["You are Orin, the Rope Seller.", "You sell a Rope for 6 coins."],
     "The player wants to buy something from you.", "6 coins, no haggling. Grab your rope and vanish."),
    ("hostile", ["You are Mira, the Healer."],
     "The player offers to help you.", "Help me? After the way you strut around here? Find another healer."),
    ("hostile", ["You are Tomas, the Clerk."],
     "The player asks you a question.", "Questions? I have no answers for the likes of you."),
    ("tense", ["You are Mira, the Healer."],
     "The player asks you a question.", "Quiet... I heard something outside. What do you want to know?"),
    ("tense", ["You are Bran, the Guard."],
     "The player greets you.", "Stay close, did you hear that? Something is moving out there."),
    ("tense", ["You are Yuri, the Hunter.", "You sell an Arrow for 2 coins."],
     "The player wants to buy something from you.", "Arrows? Take them, 2 coins each. Quickly, before it comes back."),
    ("tense", ["You are Hilda, the Baker."],
     "The player attacks you.", "No, no, back away! I will call the guards, I swear it."),
    ("neutral", ["You are Pell, the Beekeeper.", "You sell a Honey jar for 9 coins."],
     "The player wants to buy something from you.", "The honey jar is 9 coins. This year's batch came from the clover fields."),
    ("neutral", ["You are Tomas, the Clerk."],
     "The player asks you a question.", "Ask away. I have a few minutes before the ledger needs closing."),
    ("neutral", ["You are Alan, the Bartender."],
     "The player greets you.", "Evening. We have stew tonight, or ale if you prefer."),
    ("neutral", ["You are Kael, the Blacksmith."],
     "The player offers to help you.", "I can mend that. It will take an hour, and the price is fair."),
    ("neutral", ["You are Bran, the Guard."],
     "The player attacks you.", "That was a mistake. Lower your weapon and we will talk."),
)

_rng = random.Random()


_EXAMPLE_VARIANTS = 0  # 0 = fully random examples; N >= 1 = N fixed example sets per mood


def set_example_variants(n: int) -> None:
    """Bounded rotation: each mood has `n` fixed example sets and one is picked at random per call.

    Every set is an identical prompt prefix, so llama.cpp's prompt cache can reuse it (speed),
    while the choice among sets keeps replies from repeating (variety). n=1 means one fixed set.
    """
    global _EXAMPLE_VARIANTS
    _EXAMPLE_VARIANTS = max(0, int(n))


def set_stable_examples(stable: bool) -> None:
    """Shortcut: True = a single fixed set per mood, False = random examples every call."""
    set_example_variants(1 if stable else 0)


def set_example_seed(seed: Optional[int]) -> None:
    """Make the few-shot rotation reproducible (used by the eval); None re-randomizes."""
    global _rng
    _rng = random.Random(seed)


def style_line(mood: str) -> str:
    return f"Style: {MOOD_STYLE[mood]}."


def select_few_shot(mood: str, limit: int = 3, action: Optional[str] = None):
    """Few-shot examples for the current mood as (facts, action, reply).

    One example matches `action` (the narrated player action) when the pool has one, so the
    model sees the right format; the rest are random, so no single phrase dominates.
    """
    same = [e for e in _EXAMPLES if e[0] == mood] or [e for e in _EXAMPLES if e[0] == "neutral"]
    if _EXAMPLE_VARIANTS:
        v = _rng.randrange(min(_EXAMPLE_VARIANTS, len(same)))
        window = [same[(v * limit + j) % len(same)] for j in range(min(limit, len(same)))]
        return [(facts + [style_line(m)], act, reply) for m, facts, act, reply in window]
    chosen = []
    matching = [e for e in same if action and e[2] == action]
    if matching:
        chosen.append(_rng.choice(matching))
    others = [e for e in same if e not in chosen]
    chosen += _rng.sample(others, min(limit - len(chosen), len(others)))
    _rng.shuffle(chosen)
    return [(facts + [style_line(m)], act, reply) for m, facts, act, reply in chosen]


# Every example reply, used by the eval to detect verbatim copying
FEW_SHOT = tuple((facts, action, reply) for _m, facts, action, reply in _EXAMPLES)


def style_for(mood: str, initiative: float) -> str:
    """Style instruction from the mood plus the initiative/dominance axis."""
    style = MOOD_STYLE.get(mood, MOOD_STYLE["neutral"])
    if initiative >= 0.2 and mood != "hostile":
        style += "; take the initiative and ask or propose something"
    elif initiative <= -0.2:
        style += "; stay reserved and brief"
    return f"Style: {style}."


def narrate_action(action: str) -> str:
    """Rewrite what the player typed as a second-person sentence the small model can follow."""
    parsed = parse(action)
    if parsed.intent == "haggle" and parsed.offer is not None:
        return f"The player offers {parsed.offer} coins and tries to haggle over the price."
    return _INTENT_NARRATION.get(parsed.intent) or f"The player does this: {action}."


_HIDDEN_PREFIXES = ("NPC: ", "Scene: ", "Player earlier: ")  # engine bookkeeping, noise for the model


def _user_turn(facts: List[str], situation: str, no_think: bool = True) -> str:
    style = next((f for f in facts if f.startswith("Style: ")), "")
    shown = [f for f in facts if not f.startswith(("Style: ",) + _HIDDEN_PREFIXES)]
    lines = "\n".join(f"- {line}" for line in shown)
    prefix = "/no_think\n" if no_think else ""
    return f"{prefix}Facts:\n{lines}\n\n{situation}\n{style}".rstrip()


def mood_from_payload(payload: List[str]) -> str:
    """Recover the mood from the payload's Style line (the engine only sends the style)."""
    style = next((l for l in payload if l.startswith("Style: ")), "")
    return next((m for m, text in MOOD_STYLE.items() if text in style), "neutral")


def build_messages(payload: List[str], action: str, mood: Optional[str] = None,
                   no_think: bool = True, shots: int = 3) -> List[Dict[str, str]]:
    """Chat messages: system prompt, few-shot turns, then the real situation.
    `/no_think` disables Qwen3's reasoning mode. `shots=0` gives the zero-shot prompt that a
    fine-tuned model is trained and served with."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    examples = select_few_shot(mood or mood_from_payload(payload), limit=shots,
                               action=narrate_action(action)) if shots > 0 else []
    for facts, shot_action, reply in examples:
        messages.append({"role": "user", "content": _user_turn(facts, shot_action, no_think)})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": _user_turn(payload, narrate_action(action), no_think)})
    return messages


def clean_output(text: str) -> str:
    """Drop reasoning blocks and surrounding whitespace/quotes."""
    text = _THINK_RE.sub("", text).strip()
    text = re.sub(r"^(NPC[_\w]*|[A-Z][a-z]+)\s*:\s*", "", text)  # drop "Name:" prefix
    return text.strip('"').strip()


class LLMBackend:
    def generate(self, payload: List[str], action: str) -> str:
        raise NotImplementedError


class StubBackend(LLMBackend):
    """No model: echoes a deterministic line. Used by tests and as the default."""

    def generate(self, payload: List[str], action: str) -> str:
        return FALLBACK_LINES.get(mood_from_payload(payload), FALLBACK_LINES["neutral"])


class LlamaCppBackend(LLMBackend):
    """Local GGUF model through llama-cpp-python (pip install llama-cpp-python)."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, n_ctx: int = 1024,
                 n_threads: Optional[int] = None, max_tokens: int = 48,
                 temperature: float = 0.4, n_gpu_layers: int = 0, prompt_cache_mb: int = 0,
                 top_p: float = 0.8, top_k: int = 20):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("llama-cpp-python is not installed (pip install llama-cpp-python)") from e
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                          n_gpu_layers=n_gpu_layers, verbose=False)
        if prompt_cache_mb > 0:  # remember the evaluated state of earlier prompts (shared prefixes are free)
            from llama_cpp.llama_cache import LlamaRAMCache
            self._llm.set_cache(LlamaRAMCache(capacity_bytes=prompt_cache_mb * 1024 * 1024))
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p, self.top_k = top_p, top_k  # a peaked model + low top_p collapses to one answer
        self.shots = 3  # few-shot examples per call; a fine-tuned model uses 0
        meta = self._llm.metadata
        self.name = meta.get("general.name", os.path.basename(model_path))
        self._is_qwen = meta.get("general.architecture", "").lower() == "qwen3"
        # Some models (LFM2.5) open a <think> block in the generation prompt; we close it
        # right away so the reply is the NPC line and not a reasoning trace.
        self._template = meta.get("tokenizer.chat_template", "")
        self._formatter = None
        if not self._is_qwen and self._template:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
            tok = lambda t: self._llm.detokenize([t]).decode("utf-8", "ignore")
            self._formatter = Jinja2ChatFormatter(
                template=self._template, eos_token=tok(self._llm.token_eos()),
                bos_token=tok(self._llm.token_bos()), add_generation_prompt=True)

    def generate(self, payload: List[str], action: str) -> str:
        messages = build_messages(payload, action, no_think=self._is_qwen, shots=self.shots)
        sampling = dict(max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        if self._formatter is None:
            out = self._llm.create_chat_completion(messages=messages, **sampling)
            return clean_output(out["choices"][0]["message"]["content"])
        prompt = self._formatter(messages=messages).prompt
        if prompt.rstrip().endswith("<think>"):
            prompt = prompt.rstrip() + "\n\n</think>\n\n"
        out = self._llm.create_completion(prompt, stop=["<|im_end|>"], **sampling)
        return clean_output(out["choices"][0]["text"])


def load_backend(model_path: str = DEFAULT_MODEL_PATH) -> LLMBackend:
    """Best available backend: the local model if it can load, otherwise the stub."""
    try:
        return LlamaCppBackend(model_path)
    except Exception as e:
        print(f"[LLMLayer] local model unavailable ({e}); using stub backend")
        return StubBackend()
