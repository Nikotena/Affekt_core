"""The player's local hour reaches the NPC: a Time line in the payload and a greeting that fits it."""
import unittest

from AffektEngine import AffektEngine, part_of_day
from modules.LLMLayer import StubBackend, build_messages


def engine():
    e = AffektEngine(llm=StubBackend())
    e.initialize_modules()
    e.running = True
    return e


class TimeOfDayTests(unittest.TestCase):
    def test_parts_of_the_day(self):
        cases = {0: "night", 4.9: "night", 5: "morning", 11.9: "morning", 12: "afternoon",
                 17.9: "afternoon", 18: "evening", 21.9: "evening", 22: "night", 23.5: "night", 24: "night"}
        for hour, part in cases.items():
            self.assertEqual(part_of_day(hour), part, hour)

    def test_payload_has_no_time_until_the_host_sets_it(self):
        payload, _ = engine().build_payload("NPC_1", "Hello!")
        self.assertFalse(any(line.startswith("Time:") for line in payload))

    def test_payload_carries_the_time_after_set_time_of_day(self):
        e = engine()
        e.set_time_of_day(8.5)
        payload, _ = e.build_payload("NPC_1", "Hello!")
        self.assertIn("Time: morning.", payload)
        self.assertEqual(payload[1], "Time: morning.")  # right after the World line

    def test_greeting_prompt_asks_for_the_matching_greeting(self):
        expected = {9: 'Greet the player with "Good morning".', 14: 'Greet the player with "Good afternoon".',
                    19: 'Greet the player with "Good evening".',
                    23: "Greet the player and mention that it is late at night.",
                    2: "Greet the player and mention that it is late at night."}
        for hour, ask in expected.items():
            e = engine()
            e.set_time_of_day(hour)
            payload, _ = e.build_payload("NPC_1", "Hello!")
            self.assertIn(ask, build_messages(payload, "Hello!", shots=0)[-1]["content"], hour)

    def test_only_greetings_get_the_instruction(self):
        e = engine()
        e.set_time_of_day(9)
        payload, _ = e.build_payload("NPC_1", "How much for a potion?")
        user = build_messages(payload, "How much for a potion?", shots=0)[-1]["content"]
        self.assertNotIn("Greet the player with", user)
        self.assertIn("Time: morning.", user)  # the model still knows the time

    def test_no_instruction_without_a_time(self):
        payload, _ = engine().build_payload("NPC_1", "Hello!")
        self.assertNotIn("Greet the player with", build_messages(payload, "Hello!", shots=0)[-1]["content"])


if __name__ == "__main__":
    unittest.main()
