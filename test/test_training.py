"""Tests for the fine-tuning dataset pipeline (no model needed)."""
import argparse
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))

from KankeiEngine import KankeiEngine
from modules.LLMLayer import build_messages
from training.filters import drop_near_duplicates, normalize_prices, reject_reason, words_to_int
from training.generate_dataset import bench_keys, cmd_filter, cmd_review, read_jsonl, to_sample
from training.scenarios import (HELD_OUT_ITEMS, HELD_OUT_ROLES, Scenario, generate_scenarios,
                                make_payload)


def engine_messages(npc_id, mood_vector, action):
    e = KankeiEngine()
    e.initialize_modules()
    e.personality_for(npc_id).vector = list(mood_vector)
    payload, _ = e.build_payload(npc_id, action)
    return build_messages(payload, action, shots=0)


class ParityTests(unittest.TestCase):
    def test_commerce_prompt_matches_the_engine(self):
        sc = Scenario(id="x", name="Marta", role="Merchant", place="Market", mood="neutral",
                      initiative=0.0, action="buy a potion from me", kind="buy", item="Potion", price=25)
        mine = build_messages(make_payload(sc), sc.action, shots=0)
        self.assertEqual(mine, engine_messages("NPC_1", [0, 0, 0, 0.5], sc.action))

    def test_known_player_prompt_matches_the_engine(self):
        sc = Scenario(id="x", name="Alan", role="Bartender", place="Tavern", mood="friendly",
                      initiative=0.0, action="greet the NPC", kind="greet",
                      knows_player=True, knows_since="Act I")
        mine = build_messages(make_payload(sc), sc.action, shots=0)
        self.assertEqual(mine, engine_messages("NPC_0", [0.6, 0, 0.1, 0.5], sc.action))

    def test_zero_shot_has_no_examples(self):
        msgs = build_messages(["Style: x."], "greet the NPC", shots=0)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertEqual(len(build_messages(["Style: x."], "greet the NPC", shots=3)), 8)


class ScenarioTests(unittest.TestCase):
    def test_bench_scenarios_are_excluded(self):
        keys = bench_keys()
        self.assertTrue(keys)
        for sc in generate_scenarios(1500, seed=3, exclude_keys=keys):
            k = (sc.role, sc.mood, sc.kind, sc.item)
            self.assertNotIn(k, keys)
            self.assertNotIn(k[:3], keys)

    def test_mood_mix_favours_the_hard_moods(self):
        from collections import Counter
        c = Counter(s.mood for s in generate_scenarios(2000, seed=6))
        self.assertGreater(c["hostile"], c["neutral"])
        self.assertGreater(c["tense"], c["neutral"])

    def test_unseen_share_is_small(self):
        scs = generate_scenarios(2000, seed=7)
        unseen = sum(s.split == "eval_unseen" for s in scs) / len(scs)
        self.assertLess(unseen, 0.25)

    def test_held_out_roles_and_items_never_reach_train(self):
        scs = generate_scenarios(1500, seed=4)
        for sc in scs:
            if sc.role in HELD_OUT_ROLES or (sc.item in HELD_OUT_ITEMS and "coins" in " ".join(make_payload(sc))):
                self.assertEqual(sc.split, "eval_unseen")
            else:
                self.assertIn(sc.split, ("train", "eval_seen"))
        self.assertTrue(any(s.split == "eval_unseen" for s in scs))
        self.assertTrue(any(s.split == "train" for s in scs))

    def test_reproducible_and_unique(self):
        a = [s.to_dict() for s in generate_scenarios(200, seed=9)]
        b = [s.to_dict() for s in generate_scenarios(200, seed=9)]
        self.assertEqual(a, b)
        self.assertEqual(len({s["id"] for s in a}), 200)

    def test_buy_scenarios_expect_the_price(self):
        buys = [s for s in generate_scenarios(500, seed=5) if s.kind == "buy"]
        self.assertTrue(buys)
        for s in buys:
            self.assertEqual(s.expected, [str(s.price)])
            self.assertTrue(any(f"for {s.price} coins" in line for line in make_payload(s)))


class FilterTests(unittest.TestCase):
    def test_rejections(self):
        cases = {
            "*He grabs a tankard and hurls it.*": "stage_direction",
            "Get out (angrily) of my shop, fool.": "stage_direction",
            "She says nothing and leaves the room, fool.": "third_person",
            "Alan: Get out of my tavern, you fool.": "name_prefix",
            "Out.": "too_short_words",
            "The player is a fool, so get out.": "reasoning_leak",
            "I should be hostile here, so get out, fool.": "reasoning_leak",
            "Get off me, you piece of shit, fool.": "profanity",
        }
        for reply, why in cases.items():
            self.assertEqual(reject_reason(reply, "hostile", []), why, reply)

    def test_reciting_own_name_or_role_is_rejected(self):
        self.assertEqual(reject_reason("Greetings. I'm Lorcan, the Miller. What do you need?", "friendly", [],
                                       name="Lorcan", role="Miller"), "recites_identity")
        self.assertEqual(reject_reason("Welcome, friend! I am a Miller, glad to help.", "friendly", [],
                                       name="Lorcan", role="Miller"), "recites_identity")
        self.assertIsNone(reject_reason("Welcome, friend! The mill is busy today, glad you came.",
                                        "friendly", [], name="Lorcan", role="Miller"))

    def test_wrong_place_is_rejected(self):
        line = "Welcome, friend! The forge has a good fire tonight."
        self.assertEqual(reject_reason(line, "friendly", [], place="Chapel"), "wrong_place")
        self.assertIsNone(reject_reason(line, "friendly", [], place="Forge"))

    def test_invented_or_wrong_prices_are_rejected(self):
        line = "Welcome, friend! Mirrors are 2 coins each today."
        self.assertEqual(reject_reason(line, "friendly", [], price=45, price_shown=False), "invented_price")
        self.assertEqual(reject_reason(line, "friendly", [], price=45, price_shown=True), "invented_price")
        ok = "Welcome, friend! That one is 45 coins, a fair price."
        self.assertIsNone(reject_reason(ok, "friendly", [], price=45, price_shown=True))

    def test_narration_and_quoted_dialogue_are_rejected(self):
        narr = 'Isolde wipes her hands on her apron and smiles. "Ask what you will, friend."'
        self.assertEqual(reject_reason(narr, "friendly", [], name="Isolde", role="Cook"), "recites_identity")
        self.assertEqual(reject_reason('She grins. "Ask what you will, friend."', "friendly", []), "quoted_dialogue")
        self.assertEqual(reject_reason('Ask me anything, "friend", and I will answer.', "friendly", []), "quoted_dialogue")

    def test_own_name_anywhere_is_rejected(self):
        self.assertEqual(reject_reason("...Hale. The gate is closed, friend.", "tense", [], name="Hale", role="Guard"),
                         "recites_identity")

    def test_spelled_out_prices_are_rejected(self):
        for line in ("Welcome, friend! Seventy-five coins and it is yours.",
                     "Welcome, friend! That is twelve coins for the lot.",
                     "Welcome, friend! One hundred twenty gold coins, no less."):
            self.assertEqual(reject_reason(line, "friendly", []), "spelled_price", line)
        self.assertIsNone(reject_reason("Welcome, friend! That is 12 coins for the lot.", "friendly", [],
                                        price=12, price_shown=True))

    def test_profanity_can_be_allowed(self):
        line = "Get out of my shop, you fool, damn your hide."
        self.assertIsNone(reject_reason(line, "hostile", []))
        self.assertEqual(reject_reason(line.replace("damn", "fuck"), "hostile", []), "profanity")
        self.assertIsNone(reject_reason(line.replace("damn", "fuck"), "hostile", [], allow_profanity=True))

    def test_accepts_a_clean_line(self):
        self.assertIsNone(reject_reason("You dare insult me? Get out before I lose my temper.", "hostile", []))

    def test_missing_price_is_rejected(self):
        self.assertEqual(reject_reason("Take it and go away.", "neutral", ["25"]), "facts")
        self.assertEqual(reject_reason("The ring is a fine buy, truly worth it.", "neutral", ["25"]), "facts")

    def test_near_duplicates(self):
        rows = [{"reply": "Welcome, friend! Come in and sit by the fire."},
                {"reply": "Welcome, friend! Come in and sit by the fire now."},
                {"reply": "Move along before I lose my patience."}]
        self.assertEqual(len(drop_near_duplicates(rows)), 2)


class PriceNormalizationTests(unittest.TestCase):
    def test_words_to_int(self):
        cases = {"twelve": 12, "seventy-five": 75, "one hundred": 100, "one hundred twenty": 120,
                 "two hundred and fifty": 250, "Twenty": 20, "ninety-nine": 99}
        for words, n in cases.items():
            self.assertEqual(words_to_int(words), n, words)
        self.assertIsNone(words_to_int("banana"))

    def test_prices_become_digits(self):
        self.assertEqual(normalize_prices("That is twelve coins for the lot."), "That is 12 coins for the lot.")
        self.assertEqual(normalize_prices("Seventy-five gold coins, no less."), "75 gold coins, no less.")
        self.assertEqual(normalize_prices("One hundred and twenty coins."), "120 coins.")
        self.assertEqual(normalize_prices("It costs 12 coins."), "It costs 12 coins.")
        self.assertEqual(normalize_prices("Ten men came, none with coins."), "Ten men came, none with coins.")

    def test_normalized_line_passes_the_price_check(self):
        line = normalize_prices("Welcome, friend! That one is forty-five coins, a fair price.")
        self.assertIsNone(reject_reason(line, "friendly", [], price=45, price_shown=True))
        wrong = normalize_prices("Welcome, friend! That one is fifty coins, a fair price.")
        self.assertEqual(reject_reason(wrong, "friendly", [], price=45, price_shown=True), "invented_price")


class PipelineTests(unittest.TestCase):
    def test_filter_and_review_end_to_end(self):
        scs = generate_scenarios(60, seed=2, exclude_keys=bench_keys())
        with tempfile.TemporaryDirectory() as d:
            raw = os.path.join(d, "raw.jsonl")
            with open(raw, "w", encoding="utf-8") as f:
                for i, sc in enumerate(scs):
                    lead = {"hostile": "Get out, fool", "friendly": "Welcome, friend"}.get(sc.mood, "Listen")
                    reply = f"{lead}, the {sc.item.lower()} costs {sc.price} coins." if i % 3 else "*grumbles* Get out of my shop."
                    f.write(json.dumps({**sc.to_dict(), "payload": make_payload(sc), "reply": reply}) + "\n")
            cmd_filter(argparse.Namespace(raw=raw, out_dir=d))
            cmd_review(argparse.Namespace(out_dir=d, k=5, seed=1))
            for split in ("train", "eval_seen", "eval_unseen"):
                for row in read_jsonl(os.path.join(d, f"{split}.jsonl")):
                    self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])
                    self.assertIn("/no_think", row["messages"][1]["content"])
            self.assertTrue(sum(len(read_jsonl(os.path.join(d, f"{x}.jsonl")))
                                for x in ("train", "eval_seen", "eval_unseen")) > 0)
            rejected = read_jsonl(os.path.join(d, "rejected.jsonl"))
            self.assertTrue(all(r["rejected"] for r in rejected))
            self.assertTrue(os.path.isfile(os.path.join(d, "review.md")))

    def test_sample_is_zero_shot(self):
        sc = generate_scenarios(1, seed=1)[0]
        s = to_sample({**sc.to_dict(), "payload": make_payload(sc), "reply": "Fine, take it."})
        self.assertEqual(len(s["messages"]), 3)


if __name__ == "__main__":
    unittest.main()
