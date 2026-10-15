#!/usr/bin/env python3
"""Talk to the NPCs in your terminal.

    python demo.py            # uses the AMD API if AMD_API_KEY is set, else canned lines
    python demo.py --offline  # no model at all: deterministic lines only

Type a line to speak to the current NPC. Commands: /npc <name> to switch, /who to list, /quit.
"""
import argparse
import sys

from AffektEngine import create_engine
from modules.LLMLayer import StubBackend
from modules.ApiBackend import AMD_KEY_NAME, load_key


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="do not call any model")
    ap.add_argument("--llm-intents", action="store_true", help="let the model read ambiguous lines (extra API call)")
    args = ap.parse_args()

    use_api = not args.offline and bool(load_key(key_name=AMD_KEY_NAME))
    if use_api:
        engine = create_engine(use_amd_api=True, llm_intents=args.llm_intents)
    else:
        print("(no AMD_API_KEY found or --offline: NPCs answer with canned lines)")
        engine = create_engine(llm=StubBackend())

    npcs = {n.properties["name"].lower(): (k, n.properties["name"], n.properties["role"])
            for k, n in engine.graph.nodes.items() if k.startswith("NPC_")}
    current = npcs["marta"]

    def who():
        print("NPCs: " + ", ".join(f"{name} the {role}" for _, name, role in npcs.values()))

    who()
    print(f"You are talking to {current[1]} the {current[2]}.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/who":
            who()
            continue
        if line.startswith("/npc "):
            wanted = npcs.get(line[5:].strip().lower())
            if wanted:
                current = wanted
                print(f"Now talking to {current[1]} the {current[2]}.")
            else:
                who()
            continue
        r = engine.process_action(current[0], line)
        print(f"{current[1]} [{r['mood']}, {r['intent']}]> {r['response']}")
    engine.shutdown()


if __name__ == "__main__":
    main()
