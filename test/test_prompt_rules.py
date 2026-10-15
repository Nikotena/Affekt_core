"""Prompt rules found by running the player's own lines end to end."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KankeiEngine import KankeiEngine
from modules.IntentParser import is_commerce
from modules.LLMLayer import build_messages

PAYLOAD = ["You are Marta, the Merchant.", "You are at the Market.", "You sell a Potion for 25 coins.",
           "Style: neutral and businesslike."]


class CommerceTests(unittest.TestCase):
    def test_only_commercial_talk_mentions_the_goods(self):
        for text in ["How is life treating you?", "ask a question", "Hello there!", "Where is the inn?"]:
            self.assertFalse(is_commerce(text), text)
        for text in ["How much for the potion?", "I'll take the red cloak.", "purchase the sword",
                     "buy a potion from me", "What's the price of that?", "sell something"]:
            self.assertTrue(is_commerce(text), text)

    def test_small_talk_gets_no_sale_fact_but_a_purchase_does(self):
        e = KankeiEngine()
        e.initialize_modules()
        e.running = True
        chat = e.process_action("NPC_1", "How is life treating you?")
        self.assertFalse(any("for 25 coins" in line for line in chat["payload"]))
        shop = e.process_action("NPC_1", "How much for a potion?")
        self.assertTrue(any("for 25 coins" in line for line in shop["payload"]))


class RepetitionTests(unittest.TestCase):
    def test_last_reply_opening_is_forbidden_in_the_final_turn_only(self):
        facts = PAYLOAD + ["Last reply: Watch your tone, worm. I am busy."]
        msgs = build_messages(facts, "How is life treating you?", shots=2)
        self.assertIn('Do not start like your last reply ("Watch your tone,...")', msgs[-1]["content"])
        self.assertNotIn("Last reply", " ".join(m["content"] for m in msgs[:-1]))
        self.assertNotIn("worm", msgs[-1]["content"].split("Facts:")[1].split("The player")[0])  # not shown as a fact

    def test_no_instruction_without_a_previous_reply(self):
        self.assertNotIn("Do not start like", build_messages(PAYLOAD, "Hello there!", shots=0)[-1]["content"])

    def test_engine_remembers_each_npcs_last_reply(self):
        e = KankeiEngine()
        e.initialize_modules()
        e.running = True
        first = e.process_action("NPC_1", "Hello there!")
        self.assertIn("Last reply: " + first["response"], e.process_action("NPC_1", "Hello again!")["payload"])
        other = e.process_action("NPC_0", "Hello there!")
        self.assertFalse(any(line.startswith("Last reply") for line in other["payload"]))


if __name__ == "__main__":
    unittest.main()
