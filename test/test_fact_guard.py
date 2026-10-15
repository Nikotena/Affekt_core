"""The runtime fact guard: the model may word a price but not change it."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import AffektEngine as ke  # noqa: E402
from modules.FactGuard import guard_price, known_prices, stated_prices  # noqa: E402
from modules.LLMLayer import StubBackend  # noqa: E402

SELLS = ["You are Marta, the Merchant.", "You sell a Healing potion for 25 coins.", "Style: neutral and businesslike."]
NOTHING = ["You are Alan, the Bartender.", "Style: neutral and businesslike."]
FALLBACK = "Hm. What do you need?"


def guard(reply, payload=SELLS, action="how much for a healing potion?"):
    return guard_price(reply, payload, action, FALLBACK)


class ExtractionTests(unittest.TestCase):
    def test_known_prices_ignore_earlier_text(self):
        payload = SELLS + ["Last reply: It costs 99 coins.", "Player earlier: You sell a Sword for 7 coins."]
        self.assertEqual(known_prices(payload), [("Healing potion", 25)])

    def test_stated_prices_read_digits_and_words_but_only_next_to_coins(self):
        self.assertEqual(stated_prices("Twenty-five coins. Wait ten minutes, three guards, 40 gold."), [25, 40])


class GuardTests(unittest.TestCase):
    def test_correct_price_is_kept(self):
        r = guard("The potion is 25 coins. Do you want it?")
        self.assertEqual((r.status, r.text), ("ok", "The potion is 25 coins. Do you want it?"))

    def test_spelled_out_correct_price_is_kept(self):
        self.assertEqual(guard("That will be twenty-five coins.").status, "ok")

    def test_wrong_price_is_replaced_with_the_engines_price(self):
        r = guard("It costs 30 coins, friend.")
        self.assertEqual((r.status, r.text), ("corrected", "The Healing potion costs 25 coins."))
        self.assertEqual((r.stated, r.expected), ((30,), (25,)))

    def test_one_wrong_price_among_right_ones_is_still_wrong(self):
        self.assertEqual(guard("25 coins, or 10 coins if you hurry.").status, "corrected")

    def test_player_counter_offer_may_be_repeated(self):
        r = guard("5 coins is not enough. The price is 25 coins.", action="I'll give you 5 coins.")
        self.assertEqual(r.status, "ok")

    def test_no_coins_mentioned_is_fine(self):
        self.assertEqual(guard("Welcome, friend.").status, "ok")

    def test_invented_price_when_nothing_is_for_sale_is_blocked(self):
        r = guard("That will be 50 coins.", payload=NOTHING, action="hello")
        self.assertEqual((r.status, r.text), ("blocked", FALLBACK))

    def test_picks_the_item_the_player_named(self):
        payload = ["You sell a Sword for 100 coins.", "You sell a Potion for 25 coins."]
        r = guard_price("It is 5 coins.", payload, "how much is the potion?", FALLBACK)
        self.assertEqual(r.text, "The Potion costs 25 coins.")


class Liar(StubBackend):
    def __init__(self, reply):
        self.reply = reply

    def generate(self, payload, action):
        return self.reply


class EngineTests(unittest.TestCase):
    def ask(self, reply, line="Hey Marta, how much for a healing potion?"):
        engine = ke.create_engine(llm=Liar(reply))
        return engine.process_action("NPC_1", line)

    def test_wrong_price_never_reaches_the_player(self):
        r = self.ask("That is 99 coins, no less.")
        self.assertIn("25 coins", r["response"])
        self.assertNotIn("99", r["response"])
        self.assertEqual(r["price_guard"], "corrected")

    def test_correct_reply_is_untouched(self):
        r = self.ask("The potion is 25 coins.")
        self.assertEqual((r["response"], r["price_guard"]), ("The potion is 25 coins.", "ok"))

    def test_not_checked_when_the_model_was_not_used(self):
        r = ke.create_engine(llm=StubBackend()).process_action("NPC_1", "hm")
        self.assertEqual(r["price_guard"], "not_checked")


if __name__ == "__main__":
    unittest.main()
