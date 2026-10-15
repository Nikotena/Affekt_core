"""Tests for the rule-based intent parser and its use inside the engine."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))

from AffektEngine import AffektEngine
from modules.IntentParser import INTENT_DELTAS, INTENTS, IntentParser, is_commerce, parse
from modules.LLMLayer import narrate_action
from player_lines import LINES
from player_lines_heldout import HELDOUT_LINES
from player_lines_user import USER_LINES

P = IntentParser(items=["Potion", "Sword"])


class LegacyActionTests(unittest.TestCase):
    """Menu-style actions the engine used before must keep their meaning."""

    def test_old_actions_keep_their_intent(self):
        expected = {
            "greet the NPC": "greet", "say hello": "greet", "attack the NPC": "attack",
            "fight the NPC": "attack", "help the NPC": "offer_help", "offer aid": "offer_help",
            "buy a potion from me": "buy", "I want to buy the potion": "buy", "purchase the sword": "buy",
            "trade with the NPC": "sell", "sell something": "sell", "ask about the quest": "ask",
            "ask a question": "ask", "dance": "other", "stare silently": "other",
        }
        for text, intent in expected.items():
            self.assertEqual(parse(text).intent, intent, text)

    def test_original_deltas_are_unchanged(self):
        self.assertEqual(INTENT_DELTAS["greet"], [0.3, -0.2, 0.1, 0.7])
        self.assertEqual(INTENT_DELTAS["attack"], [-0.4, 0.3, 0.8, 0.9])
        self.assertEqual(INTENT_DELTAS["offer_help"], [0.4, -0.1, 0.2, 0.6])


class ParsingTests(unittest.TestCase):
    def test_priority_threat_beats_compliment_and_purchase_beats_greeting(self):
        self.assertEqual(parse("Nice shop. Shame if something happened to it.").intent, "threaten")
        self.assertEqual(parse("Hello, I'd like to buy a potion.").intent, "buy")

    def test_negated_violence_is_not_an_attack(self):
        self.assertEqual(parse("I won't fight you. I come in peace.").intent, "greet")
        self.assertEqual(parse("I'm not here to hurt anyone.").intent, "other")
        self.assertEqual(parse("Please don't hurt me!").intent, "ask_help")

    def test_keyword_traps(self):
        self.assertEqual(parse("I'm afraid I can't pay that much.").intent, "haggle")  # 'afraid' has 'aid'
        self.assertEqual(parse("That's a hard task for one person.").intent, "other")   # 'task' has 'ask'
        self.assertEqual(parse("Don't make me ask twice.").intent, "threaten")

    def test_slang_is_normalised_without_touching_normal_words(self):
        self.assertEqual(parse("u r an idiot").intent, "insult")
        self.assertEqual(parse("Gimme a potion.").intent, "buy")
        self.assertEqual(parse("I'm gonna kill you.").intent, "threaten")
        self.assertEqual(parse("Where can I find the blacksmith?").intent, "ask")  # no stray u/r rewriting

    def test_offer_and_item_extraction(self):
        p = P.parse("Will you sell me that potion for 20 coins?")
        self.assertEqual((p.intent, p.offer, p.item), ("haggle", 20, "Potion"))
        self.assertIsNone(P.parse("Give me two arrows please").offer)
        self.assertEqual(P.parse("I want the sword").item, "Sword")

    def test_greeting_followed_by_a_question_is_a_question(self):
        self.assertEqual(parse("Hi! How much for the sword?").intent, "ask")
        self.assertEqual(parse("Hey there! What's up?").intent, "greet")

    def test_valence_signs(self):
        for text, sign in [("Die, you wretch!", -1), ("Thanks, you saved my life.", 1), ("Hmm.", 0),
                           ("Where is the inn?", 0), ("I'll take the red cloak.", 0)]:
            self.assertEqual(parse(text).valence, sign, text)

    def test_never_crashes_and_returns_a_known_intent(self):
        for text in ["", "   ", "???", "a" * 5000, "!!!!", "’’", "12345", None]:
            self.assertIn(parse(text).intent, INTENTS)

    def test_deltas_are_copies(self):
        p = parse("greet the NPC")
        p.delta[0] = 99
        self.assertEqual(INTENT_DELTAS["greet"][0], 0.3)


class BenchmarkRegressionTests(unittest.TestCase):
    """The parser must not fall back to keyword-level quality on realistic lines."""

    def accuracy(self, lines):
        ok = sum(parse(t).intent == i for t, i, _v in lines)
        return ok / len(lines)

    def test_dev_and_heldout_accuracy_floor(self):
        self.assertGreaterEqual(self.accuracy(LINES), 0.95)
        self.assertGreaterEqual(self.accuracy(HELDOUT_LINES), 0.90)

    def test_no_line_gets_the_opposite_feeling(self):
        for t, _i, v in LINES + HELDOUT_LINES:
            self.assertGreaterEqual(parse(t).valence * v, 0, t)


class RealPlayerLinesTests(unittest.TestCase):
    """Aaron's own lines (labels by Claude). Fixed classes of problems, not single phrases."""

    def test_politeness_openers_and_placeholders_are_ignored(self):
        self.assertEqual(parse("Excuse me, could you sign my petition?").intent, "ask")
        self.assertEqual(parse("Excuse me, [NPC Name], how much do you sell healing potions for?").intent, "buy")
        self.assertEqual(parse("Well met, innkeeper.").intent, "greet")  # 'well' is only an opener before a comma

    def test_greeting_with_a_comma_then_a_question_is_a_question(self):
        self.assertEqual(parse("Hey, did you hear what happened to the miller?").intent, "ask")
        self.assertEqual(parse("Hello there!").intent, "greet")

    def test_frustration_and_challenges_are_complaints_not_requests(self):
        for text in ["Damn it, when are you going to help me get out of this?",
                     "Who the hell do you think you are?", "Oh, so NOW that affects me",
                     "Good God, is anyone actually working here or what?"]:
            p = parse(text)
            self.assertEqual((p.intent, p.valence), ("complain", -1), text)

    def test_insult_words_work_without_you(self):
        self.assertEqual(parse("Listen, dumbass, are you gonna help me or not?").intent, "insult")
        self.assertEqual(parse("Hey stupid, get over here!").valence, -1)

    def test_needing_help_is_not_pleasant(self):
        self.assertEqual(parse("Can you help me? I'm lost.").valence, 0)

    def test_no_user_line_gets_the_opposite_feeling_and_accuracy_floor(self):
        for t, _i, v in USER_LINES:
            self.assertGreaterEqual(parse(t).valence * v, 0, t)
        acc = sum(parse(t).intent == i for t, i, _v in USER_LINES) / len(USER_LINES)
        self.assertGreaterEqual(acc, 0.8)


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.e = AffektEngine()
        self.e.initialize_modules()
        self.e.running = True

    def test_veiled_threat_makes_the_npc_hostile_not_happy(self):
        r = self.e.process_action("NPC_0", "Nice shop. Shame if something happened to it.")
        self.assertEqual(r["intent"], "threaten")
        self.assertEqual(r["mood"], "hostile")

    def test_compliment_warms_the_npc_and_neutral_text_does_not(self):
        r = self.e.process_action("NPC_1", "You run the best shop in town.")
        self.assertEqual((r["intent"], r["mood"]), ("compliment", "friendly"))
        r = self.e.process_action("NPC_3", "Hmm.")
        self.assertEqual((r["intent"], r["mood"]), ("other", "neutral"))

    def test_free_text_purchase_shows_the_price_to_the_model(self):
        r = self.e.process_action("NPC_1", "I'll take the red cloak.")
        self.assertEqual(r["intent"], "buy")
        self.assertTrue(any("for 25 coins" in line for line in r["payload"]))

    def test_haggle_offer_is_reported_and_narrated(self):
        r = self.e.process_action("NPC_1", "Will you sell me a potion for 10 coins?")
        self.assertEqual((r["intent"], r["offer"]), ("haggle", 10))
        self.assertIn("10 coins", narrate_action("Will you sell me a potion for 10 coins?"))

    def test_commerce_rule_is_shared(self):
        self.assertTrue(is_commerce("purchase the potion"))
        self.assertTrue(is_commerce("I'll take the red cloak."))
        self.assertFalse(is_commerce("Nice weather."))


if __name__ == "__main__":
    unittest.main()
