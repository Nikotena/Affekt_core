#!/usr/bin/env python3
"""A scripted 2-minute tour of the engine, for demos.

    python scene.py                      # live: AMD API if AMD_API_KEY is set, else canned lines
    python scene.py --offline            # deterministic lines only, no network
    python scene.py --save run.json      # live run, saved so it can be replayed
    python scene.py --replay run.json    # replay a saved run with no network (demo safety net)
"""
import argparse
import json
import sys
import time

from AffektEngine import create_engine
from modules.LLMLayer import StubBackend
from modules.OpenRouterBackend import AMD_KEY_NAME, load_key

# (npc_id, what the player says, what to point out)
SCRIPT = [
    ("NPC_1", "Hey Marta, how much for a healing potion?",
     "The engine knows what Marta sells and at what price; the model only phrases it."),
    ("NPC_1", "That's too expensive. I'll give you 5 coins.",
     "A price offer is read as haggling."),
    ("NPC_0", "This ale tastes like dirt, you idiot!",
     "Alan is easy-going: the insult barely moves him."),
    ("NPC_4", "This sword is garbage, you idiot!",
     "Same insult, but Kael the blacksmith is choleric: he turns hostile."),
    ("NPC_4", "Sorry, I didn't mean it. You're amazing!",
     "Kael holds a grudge: one kind word does not fix it."),
    ("NPC_4", "Thank you, really. Could you repair my sword?",
     "A few more kind turns and his mood recovers."),
    ("NPC_2", "Hello.", "Now greeting the guard, over and over..."),
] + [("NPC_2", "Hello.", "...")] * 5 + [
    ("NPC_2", "Hello.", "The seventh hello: the guard gets nervous, with good reason."),
    ("NPC_3", "I'm hurt, please help me!",
     "Elena the healer reads a request for help."),
]


def show(i, npc, name, role, line, note, r, pause):
    print(f"\n[{i}/{len(SCRIPT)}] {note}")
    print(f"  you -> {name} ({role}): {line}")
    print(f"  {name} [{r['mood']}, {r['intent']}]: {r['response']}")
    sys.stdout.flush()
    time.sleep(pause)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--save", metavar="FILE")
    ap.add_argument("--replay", metavar="FILE")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between turns")
    args = ap.parse_args()

    if args.replay:
        with open(args.replay, encoding="utf-8") as f:
            turns = json.load(f)
        for i, t in enumerate(turns, 1):
            show(i, t["npc"], t["name"], t["role"], t["line"], t["note"], t, args.pause)
        return

    use_api = not args.offline and bool(load_key(key_name=AMD_KEY_NAME))
    engine = create_engine(use_amd_api=True) if use_api else create_engine(llm=StubBackend())
    if use_api and args.save:  # a recording should hold real model lines: wait longer than a live game would
        engine.llm.timeout, engine.llm.retries = 120.0, 4
    print("Model: Gemma-4-31B-IT on AMD Token Factory" if use_api else "Model: none (canned lines)")

    turns = []
    for i, (npc, line, note) in enumerate(SCRIPT, 1):
        node = engine.graph.get_node(npc).properties
        r = engine.process_action(npc, line)
        turn = {"npc": npc, "name": node["name"], "role": node["role"], "line": line, "note": note,
                "mood": r["mood"], "intent": r["intent"], "response": r["response"]}
        turns.append(turn)
        show(i, npc, node["name"], node["role"], line, note, turn, args.pause)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(turns, f, ensure_ascii=False, indent=1)
        print(f"\nSaved to {args.save}; replay with: python scene.py --replay {args.save}")
    engine.shutdown()


if __name__ == "__main__":
    main()
