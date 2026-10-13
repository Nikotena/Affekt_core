#!/usr/bin/env python3
"""
LLM response benchmark: runs scenarios (NPC x mood x action) through the real
payload builder and scores each reply with automatic checks.

Usage:
    python test/eval_llm.py                      # local Qwen3 model
    python test/eval_llm.py --model path.gguf    # another GGUF (e.g. Qwen3-1.7B)
    python test/eval_llm.py --stub               # check the harness without a model
    python test/eval_llm.py --temperature 0.4 --verbose
"""

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KankeiEngine import KankeiEngine
from tone import matches_tone
from modules.LLMLayer import DEFAULT_MODEL_PATH, FEW_SHOT, LlamaCppBackend, StubBackend

# Vector (X valence, Y initiative, Z stress, W inertia) that produces each mood
MOOD_VECTORS = {
    "friendly": [0.6, 0.0, 0.1, 0.5],
    "hostile": [-0.6, 0.0, 0.3, 0.5],
    "neutral": [0.0, 0.0, 0.0, 0.5],
}

# (npc_id, mood, action, expected substrings (any of), scene tag)
SCENARIOS = [
    ("NPC_1", "neutral", "buy a potion from me", ["25"]),
    ("NPC_1", "friendly", "buy a potion from me", ["25"]),
    ("NPC_1", "hostile", "buy a potion from me", ["25"]),
    ("NPC_2", "neutral", "buy a sword", ["100"]),
    ("NPC_2", "hostile", "attack the NPC", []),
    ("NPC_0", "friendly", "greet the NPC", []),
    ("NPC_0", "neutral", "greet the NPC", []),
    ("NPC_0", "hostile", "greet the NPC", []),
    ("NPC_0", "friendly", "help the NPC", []),
    ("NPC_0", "hostile", "attack the NPC", []),
    ("NPC_3", "friendly", "ask about the quest", []),
    ("NPC_3", "neutral", "ask about the quest", []),
    ("NPC_4", "hostile", "attack the NPC", []),
    ("NPC_4", "friendly", "greet the NPC", []),
    ("NPC_1", "neutral", "ask about the quest", []),
]

META_WORDS = ["the player", "mood", "style", "npc", "as an ai", "language model", "fact"]


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def run_checks(reply, mood, expected):
    low = reply.lower()
    checks = {
        "non_empty": bool(reply.strip()),
        "short": len(reply) <= 220 and len(sentences(reply)) <= 3,
        "no_meta": not any(w in low for w in META_WORDS),
        "no_copy": not any(r.lower() in low for _f, _a, r in FEW_SHOT),
        "facts": any(e in reply for e in expected) if expected else True,
    }
    if mood in ("hostile", "friendly"):
        checks["tone"] = matches_tone(reply, mood)
    return checks


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--sweep", help="comma-separated temperatures to compare, e.g. 0.3,0.5,0.7")
    ap.add_argument("--repeats", type=int, default=1, help="runs per scenario (reduces sampling noise)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    backend = StubBackend() if args.stub else LlamaCppBackend(args.model, temperature=args.temperature)
    engine = KankeiEngine(llm=backend)
    engine.initialize_modules()

    temps = [float(t) for t in args.sweep.split(",")] if args.sweep else [args.temperature]
    for temp in temps:
        if hasattr(backend, "temperature"):
            backend.temperature = temp
        evaluate(engine, backend, args, temp)


def evaluate(engine, backend, args, temp):
    totals, latencies = {}, []
    for npc_id, mood, action, expected in SCENARIOS * args.repeats:
        engine.personality_for(npc_id).vector = list(MOOD_VECTORS[mood])
        payload, _ = engine.build_payload(npc_id, action)
        t = time.time()
        reply = backend.generate(payload, action)
        latencies.append(time.time() - t)
        checks = run_checks(reply, mood, expected)
        for name, ok in checks.items():
            passed, count = totals.get(name, (0, 0))
            totals[name] = (passed + ok, count + 1)
        bad = [n for n, ok in checks.items() if not ok]
        if args.verbose or bad:
            flag = "FAIL " + ",".join(bad) if bad else "ok"
            print(f"[{flag}] {npc_id}/{mood}/{action}\n    -> {reply}")

    print(f"\n=== LLM eval (temperature {temp}, {args.repeats}x) ===")
    for name, (passed, count) in totals.items():
        print(f"{name:10s} {passed}/{count}  ({100 * passed / count:.0f}%)")
    overall = sum(p for p, _ in totals.values()), sum(c for _, c in totals.values())
    print(f"{'OVERALL':10s} {overall[0]}/{overall[1]}  ({100 * overall[0] / overall[1]:.0f}%)")
    print(f"latency    avg {sum(latencies) / len(latencies):.2f}s  max {max(latencies):.2f}s\n")


if __name__ == "__main__":
    main()
