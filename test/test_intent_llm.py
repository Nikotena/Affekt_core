"""IntentLLM: the model only resolves "other"; failures keep the parser's answer."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.IntentLLM import parse_label, refine  # noqa: E402
from modules.IntentParser import parse  # noqa: E402


class Fake:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def complete(self, messages, max_tokens=None, temperature=None):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class IntentLLMTests(unittest.TestCase):
    def test_parse_label_prefers_the_longest_name(self):
        self.assertEqual(parse_label("ask_help"), "ask_help")
        self.assertEqual(parse_label("Label: complain."), "complain")
        self.assertIsNone(parse_label("no idea"))

    def test_model_is_not_called_when_rules_understood(self):
        fake = Fake("insult")
        self.assertEqual(refine(parse("Hello there!"), fake).intent, "greet")
        self.assertEqual(fake.calls, 0)

    def test_model_resolves_other(self):
        fake = Fake("complain")
        out = refine(parse("The bard only knows one tune."), fake)
        self.assertEqual((out.intent, out.rule, fake.calls), ("complain", "llm", 1))
        self.assertLess(out.delta[0], 0)  # the emotional push follows the new intent

    def test_failure_or_garbage_keeps_the_parser_answer(self):
        for answer in (RuntimeError("HTTP 503"), "banana", "other"):
            out = refine(parse("The bard only knows one tune."), Fake(answer))
            self.assertEqual(out.intent, "other")


class EngineIntegrationTests(unittest.TestCase):
    def engine(self, **kw):
        import AffektEngine as ke
        from modules.LLMLayer import StubBackend
        return ke.create_engine(llm=StubBackend(), **kw)

    def test_off_by_default(self):
        fake = Fake("complain")
        e = self.engine()
        self.assertIsNone(e.intent_llm)
        self.assertEqual(e.process_action("NPC_1", "The bard only knows one tune.")["intent"], "other")
        self.assertEqual(fake.calls, 0)

    def test_enabled_resolves_other_and_changes_the_mood_push(self):
        fake = Fake("complain")
        e = self.engine()
        e.intent_llm = fake
        r = e.process_action("NPC_1", "The bard only knows one tune.")
        self.assertEqual((r["intent"], fake.calls), ("complain", 1))
        e.process_action("NPC_1", "Hello there!")  # rules understood it: no extra call
        self.assertEqual(fake.calls, 1)

    def test_api_down_keeps_working(self):
        e = self.engine()
        e.intent_llm = Fake(RuntimeError("HTTP 503"))
        self.assertEqual(e.process_action("NPC_1", "The bard only knows one tune.")["intent"], "other")

    def test_create_engine_flag_reuses_a_backend_with_complete(self):
        import AffektEngine as ke
        fake = Fake("ask")
        fake.generate = lambda payload, action: "Hm."
        e = ke.create_engine(llm=fake, llm_intents=True)
        self.assertIs(e.intent_llm, fake)


if __name__ == "__main__":
    unittest.main()
