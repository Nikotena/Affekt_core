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

from AffektEngine import AffektEngine
from tone import matches_tone
from modules.TextNumbers import normalize_prices
from modules.ApiBackend import AMD_DEFAULT_MODEL
from modules.LLMLayer import (DEFAULT_MODEL_PATH, FEW_SHOT, LlamaCppBackend, StubBackend,
                              set_example_seed, set_stable_examples, set_example_variants)

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


# Words too common to count as "copying" when they make up most of a 4-word run
_STOPWORDS = set("""you i the a an to of and what do want i'll your my in is it for me on with
that this are be if at as can will from have we they he she or but not so just here there""".split())


def _shingles(text, n=4):
    words = re.findall(r"[a-z']+", text.lower().replace("’", "'"))
    runs = {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
    return {r for r in runs if sum(w in _STOPWORDS for w in r) < 3}  # drop generic runs


_EXAMPLE_SHINGLES = set().union(*(_shingles(reply) for _f, _a, reply in FEW_SHOT))


def leaks_example(reply):
    """True if the reply reuses a 4-word run from a few-shot example reply."""
    return bool(_shingles(reply) & _EXAMPLE_SHINGLES)


def run_checks(reply, mood, expected):
    low = reply.lower()
    checks = {
        "non_empty": bool(reply.strip()),
        "short": len(reply) <= 220 and len(sentences(reply)) <= 3,
        "no_meta": not any(re.search(r"\b" + re.escape(w) + r"s?\b", low) for w in META_WORDS),
        "no_copy": not any(r.lower() in low for _f, _a, r in FEW_SHOT),
        "no_leak": not leaks_example(reply),
        "facts": any(e in normalize_prices(reply) for e in expected) if expected else True,  # "twenty-five coins" counts
    }
    if mood in ("hostile", "friendly"):
        checks["tone"] = matches_tone(reply, mood)
    return checks


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--amd", nargs="?", const=AMD_DEFAULT_MODEL, metavar="MODEL",
                    help="use the AMD Token Factory API (default MiMo-V2.6-Flash; key AMD_API_KEY in .env)")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--sweep", help="comma-separated temperatures to compare, e.g. 0.3,0.5,0.7")
    ap.add_argument("--repeats", type=int, default=1, help="runs per scenario (reduces sampling noise)")
    ap.add_argument("--seed", type=int, default=None, help="seed for the few-shot rotation")
    ap.add_argument("--stable", action="store_true", help="same few-shot examples every call (enables prompt reuse)")
    ap.add_argument("--cache-mb", type=int, default=0, help="llama.cpp prompt-prefix cache size in MB")
    ap.add_argument("--variants", type=int, default=0, help="N fixed example sets per mood, rotated at random")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    set_stable_examples(args.stable)
    if args.variants:
        set_example_variants(args.variants)
    if args.amd:
        from modules.ApiBackend import AMDBackend
        backend = AMDBackend(args.amd, temperature=args.temperature)
    else:
        backend = StubBackend() if args.stub else LlamaCppBackend(
            args.model, temperature=args.temperature, n_threads=args.threads, prompt_cache_mb=args.cache_mb,
            top_p=args.top_p, top_k=args.top_k)
    if hasattr(backend, "shots"):
        backend.shots = args.shots
    engine = AffektEngine(llm=backend)
    engine.initialize_modules()
    set_example_seed(args.seed)

    temps = [float(t) for t in args.sweep.split(",")] if args.sweep else [args.temperature]
    for temp in temps:
        if hasattr(backend, "temperature"):
            backend.temperature = temp
        evaluate(engine, backend, args, temp)


def evaluate(engine, backend, args, temp):
    totals, latencies, by_scenario = {}, [], {}
    for npc_id, mood, action, expected in SCENARIOS * args.repeats:
        engine.personality_for(npc_id).vector = list(MOOD_VECTORS[mood])
        payload, _ = engine.build_payload(npc_id, action)
        t = time.time()
        reply = backend.generate(payload, action)
        latencies.append(time.time() - t)
        by_scenario.setdefault((npc_id, mood, action), []).append(reply.strip().lower())
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
    total = sum(len(v) for v in by_scenario.values())
    unique = sum(len(set(v)) for v in by_scenario.values())
    print(f"variety    {unique}/{total}  ({100 * unique / total:.0f}% distinct replies per scenario)")
    print(f"latency    avg {sum(latencies) / len(latencies):.2f}s  max {max(latencies):.2f}s\n")


if __name__ == "__main__":
    main()
