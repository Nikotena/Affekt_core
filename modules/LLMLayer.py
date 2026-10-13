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

# Few-shot examples: (facts, player action, NPC line). Short, first person, in character.
FEW_SHOT = (
    (["You are Alan, the Bartender. Your mood: friendly", "Alan KNOWS Player"],
     "The player greets you.", "Well, look who's back! The first round is on me."),
    (["You are Kael, the Blacksmith. Your mood: hostile"],
     "The player attacks you.", "You dare raise a hand in my forge? Out, before I lose my temper!"),
)


def narrate_action(action: str) -> str:
    """Rewrite a raw player action as a second-person sentence the small model can follow."""
    text = action.lower()
    for keywords, sentence in _ACTION_NARRATION:
        if any(k in text for k in keywords):
            return sentence
    return f"The player does this: {action}."


def _user_turn(facts: List[str], situation: str) -> str:
    lines = "\n".join(f"- {line}" for line in facts)
    return f"/no_think\nFacts:\n{lines}\n\n{situation}"


def build_messages(payload: List[str], action: str) -> List[Dict[str, str]]:
    """Chat messages: system prompt, few-shot turns, then the real situation.
    `/no_think` disables Qwen3's reasoning mode."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for facts, shot_action, reply in FEW_SHOT:
        messages.append({"role": "user", "content": _user_turn(facts, shot_action)})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": _user_turn(payload, narrate_action(action))})
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
        mood = next((l.rsplit("mood: ", 1)[1] for l in payload if "mood: " in l), "neutral")
        return FALLBACK_LINES.get(mood, FALLBACK_LINES["neutral"])


class LlamaCppBackend(LLMBackend):
    """Local GGUF model through llama-cpp-python (pip install llama-cpp-python)."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, n_ctx: int = 1024,
                 n_threads: Optional[int] = None, max_tokens: int = 48,
                 temperature: float = 0.7, n_gpu_layers: int = 0):
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

    def generate(self, payload: List[str], action: str) -> str:
        out = self._llm.create_chat_completion(
            messages=build_messages(payload, action),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.8,
            top_k=20,
        )
        return clean_output(out["choices"][0]["message"]["content"])


def load_backend(model_path: str = DEFAULT_MODEL_PATH) -> LLMBackend:
    """Best available backend: the local model if it can load, otherwise the stub."""
    try:
        return LlamaCppBackend(model_path)
    except Exception as e:
        print(f"[LLMLayer] local model unavailable ({e}); using stub backend")
        return StubBackend()
