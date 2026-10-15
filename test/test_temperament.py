"""Each NPC reacts in its own way to the same player lines."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import AffektEngine as ke  # noqa: E402
from modules.LLMLayer import StubBackend  # noqa: E402

NPCS = {"Alan": "NPC_0", "Marta": "NPC_1", "Guard": "NPC_2", "Elena": "NPC_3", "Kael": "NPC_4"}


def moods(npc, lines):
    e = ke.create_engine(llm=StubBackend())
    return [e.process_action(npc, t)["mood"] for t in lines]


class TemperamentTests(unittest.TestCase):
    def hostile_turns(self, npc):
        seq = moods(npc, ["You idiot!"] * 3 + ["Thank you, you're amazing!"] * 4)
        return seq.count("hostile")

    def test_grudge_holder_stays_angry_longer_than_easy_going(self):
        self.assertGreater(self.hostile_turns(NPCS["Kael"]), self.hostile_turns(NPCS["Marta"]))
        self.assertGreater(self.hostile_turns(NPCS["Marta"]), self.hostile_turns(NPCS["Alan"]))

    def test_everyone_calms_down_eventually(self):
        for name, npc in NPCS.items():
            self.assertNotEqual(moods(npc, ["You idiot!"] * 3 + ["Thank you, you're amazing!"] * 6)[-1],
                                "hostile", name)

    def test_phlegmatic_guard_shrugs_off_one_insult(self):
        self.assertNotEqual(moods(NPCS["Guard"], ["You idiot, shut up!"])[0], "hostile")
        self.assertEqual(moods(NPCS["Kael"], ["You idiot, shut up!"])[0], "hostile")

    def test_threats_still_anger_the_easy_going_bartender(self):
        self.assertEqual(moods(NPCS["Alan"], ["Nice shop. Shame if something happened to it."])[0], "hostile")

    def test_unknown_npc_uses_the_normal_temperament(self):
        e = ke.create_engine(llm=StubBackend())
        self.assertEqual(e.personality_for("NPC_99").temperament, ke.NORMAL)


if __name__ == "__main__":
    unittest.main()
