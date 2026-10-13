"""
LLM Layer (the "20" of the 80/20 split)

Turns the deterministic payload into natural language. The model never changes
engine state: it only phrases what the deterministic core already decided.
Target model: Qwen3-0.6B (GGUF, Q8_0) running locally through llama-cpp-python.
"""

import os
import re
from typing import Dict, List, Optional

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
_ACTION_NARRATION = (
    (("attack", "fight"), "The player attacks you."),
    (("greet", "hello"), "The player greets you."),
    (("help", "aid"), "The player offers to help you."),
    (("buy", "purchase"), "The player wants to buy something from you."),
    (("sell", "trade"), "The player wants to trade with you."),
    (("ask", "question"), "The player asks you a question."),
)

# The deterministic core decides HOW the NPC sounds; the model only words it.
MOOD_STYLE = {
    "friendly": "warm and welcoming; offer something",
    "hostile": "openly hostile, rude and menacing; insult or threaten the player; never polite",
    "tense": "nervous and wary; short, guarded sentences",
    "neutral": "neutral and businesslike",
}

# Few-shot examples: (mood, facts, player action, NPC line). Short, first person, in character.
_EXAMPLES = (
    ("friendly", ["You are Alan, the Bartender.", "You know the player."],
     "The player greets you.", "Look who's back! Sit down, I'll pour you something."),
    ("friendly", ["You are Dara, the Fisherman.", "You sell a Fish for 3 coins."],
     "The player wants to buy something from you.", "Of course, friend! That is 3 coins, and I'll pick the best one."),
    ("hostile", ["You are Kael, the Blacksmith."],
     "The player attacks you.", "You dare raise a hand in my forge? Out, before I lose my temper!"),
    ("hostile", ["You are Bran, the Guard."],
     "The player greets you.", "Don't smile at me, stranger. Move along before I lose patience."),
    ("hostile", ["You are Dara, the Fisherman.", "You sell a Fish for 3 coins."],
     "The player wants to buy something from you.", "It costs 3 coins. Pay up and get out of my sight."),
    ("tense", ["You are Mira, the Healer."],
     "The player asks you a question.", "Quiet... I heard something outside. What do you want to know?"),
    ("neutral", ["You are Dara, the Fisherman.", "You sell a Fish for 3 coins."],
     "The player wants to buy something from you.", "A fish is 3 coins."),
    ("neutral", ["You are Tomas, the Clerk."],
     "The player asks you a question.", "Ask away. I have a few minutes."),
)

def style_line(mood: str) -> str:
    return f"Style: {MOOD_STYLE[mood]}."


def select_few_shot(mood: str, limit: int = 2):
    """Examples for the current mood (they teach the tone), as (facts, action, reply)."""
    same = [e for e in _EXAMPLES if e[0] == mood][:limit]
    chosen = same or [e for e in _EXAMPLES if e[0] == "neutral"][:limit]
    return [(facts + [style_line(m)], action, reply) for m, facts, action, reply in chosen]


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
    """Rewrite a raw player action as a second-person sentence the small model can follow."""
    text = action.lower()
    for keywords, sentence in _ACTION_NARRATION:
        if any(k in text for k in keywords):
            return sentence
    return f"The player does this: {action}."


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
                   no_think: bool = True) -> List[Dict[str, str]]:
    """Chat messages: system prompt, few-shot turns, then the real situation.
    `/no_think` disables Qwen3's reasoning mode."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for facts, shot_action, reply in select_few_shot(mood or mood_from_payload(payload)):
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
                 temperature: float = 0.4, n_gpu_layers: int = 0):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("llama-cpp-python is not installed (pip install llama-cpp-python)") from e
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                          n_gpu_layers=n_gpu_layers, verbose=False)
        self.max_tokens = max_tokens
        self.temperature = temperature
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
        messages = build_messages(payload, action, no_think=self._is_qwen)
        sampling = dict(max_tokens=self.max_tokens, temperature=self.temperature, top_p=0.8, top_k=20)
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
