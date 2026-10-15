"""The player may address an NPC by name or nickname ([NPC_Name] in the player lines)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from AffektEngine import AffektEngine
from modules.IntentParser import IntentParser, parse, strip_vocatives

NAMES = ["Marta", "Guard", "Big Al"]


class VocativeTests(unittest.TestCase):
    def test_name_in_each_position(self):
        cases = {
            "Hey Marta, how much for the potion?": ("Hey, how much for the potion?", "Marta"),
            "Marta, I'll take the red cloak.": ("I'll take the red cloak.", "Marta"),
            "Excuse me, Marta, how much do you sell potions for?": ("Excuse me, how much do you sell potions for?", "Marta"),
            "You said it, Marta, you said it.": ("You said it, you said it.", "Marta"),
            "Thanks, Big Al.": ("Thanks.", "Big Al"),
            "marta, hello": ("hello", "marta"),
        }
        for text, expected in cases.items():
            self.assertEqual(strip_vocatives(text, NAMES), expected, text)

    def test_name_used_as_a_noun_is_left_alone(self):
        self.assertEqual(strip_vocatives("I punch the Guard in the face.", NAMES)[0], "I punch the Guard in the face.")
        self.assertEqual(parse("I punch the Guard in the face.", NAMES).intent, "attack")

    def test_bracket_placeholders_are_vocatives(self):
        text, who = strip_vocatives("Damn it, [NPC_name], when are you going to help me?", [])
        self.assertEqual(text, "Damn it, when are you going to help me?")
        self.assertIsNone(who)  # a placeholder is not a real name
        self.assertEqual(strip_vocatives("[NPC_Name], hello!", [])[0], "hello!")

    def test_meaning_is_the_same_with_or_without_the_name(self):
        for plain, named in [("How much for the potion?", "Hey Marta, how much for the potion?"),
                             ("I'll take the red cloak.", "Marta, I'll take the red cloak."),
                             ("Die, you wretch!", "Die, you wretch, Marta!"),
                             ("Thanks, you saved my life.", "Thanks, Big Al, you saved my life.")]:
            a, b = parse(plain, NAMES), parse(named, NAMES)
            self.assertEqual((a.intent, a.valence), (b.intent, b.valence), named)

    def test_nothing_to_strip(self):
        self.assertEqual(strip_vocatives("Hello there!", []), ("Hello there!", None))
        self.assertEqual(strip_vocatives("", NAMES), ("", None))


class EngineNameTests(unittest.TestCase):
    def setUp(self):
        self.e = AffektEngine()
        self.e.initialize_modules()
        self.e.running = True

    def test_engine_knows_the_npc_name_and_nicknames(self):
        self.assertEqual(self.e.names_for("NPC_1"), ["Marta"])
        self.e.add_alias("NPC_1", "Big M")
        self.e.add_alias("NPC_1", "big m")  # no duplicates
        self.assertEqual(self.e.names_for("NPC_1"), ["Marta", "Big M"])
        self.assertEqual(self.e.names_for("nobody"), [])

    def test_addressing_by_name_or_nickname(self):
        self.e.add_alias("NPC_1", "Big M")
        r = self.e.process_action("NPC_1", "Marta, I'll take the red cloak.")
        self.assertEqual((r["intent"], r["addressed"]), ("buy", "Marta"))
        self.assertEqual(r["action"], "Marta, I'll take the red cloak.")  # the player's words are kept as typed
        r = self.e.process_action("NPC_1", "Thanks, Big M. You saved my day.")
        self.assertEqual((r["intent"], r["addressed"]), ("compliment", "Big M"))

    def test_a_name_never_changes_the_prompt_the_model_sees(self):
        a = self.e.process_action("NPC_1", "How much for the potion?")
        b = self.e.process_action("NPC_1", "Hey Marta, how much for the potion?")
        self.assertEqual(a["intent"], b["intent"])

    def test_npc_named_like_an_object(self):
        r = self.e.process_action("NPC_2", "I punch the Guard in the face.")
        self.assertEqual((r["intent"], r["addressed"]), ("attack", None))

    def test_small_talk_reaches_the_model_but_a_grunt_does_not(self):
        self.assertTrue(self.e.process_action("NPC_0", "The harvest looks poor this year.")["llm_used"])
        self.assertFalse(self.e.process_action("NPC_0", "Hmm.")["llm_used"])


if __name__ == "__main__":
    unittest.main()
