"""Behavioural unit tests. Run: python -m unittest discover -s test -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AffektEngine import AffektEngine
from modules.ContextFunction import ContextFunction
from modules.GraphDatabase import GraphDatabase
from modules.LLMLayer import build_messages, clean_output, narrate_action
from modules.MemorySet import MemorySet
from modules.VectorialPersonality import VectorialPersonality, CHOLERIC


def make_engine():
    e = AffektEngine()
    e.initialize_modules()
    e.running = True
    return e


class PersonalityTests(unittest.TestCase):
    def test_action_moves_vector_and_does_not_mutate_input(self):
        p = VectorialPersonality("n")
        delta = [0.3, -0.2, 0.1, 0.7]
        p.update_from_action(delta)
        self.assertEqual(delta, [0.3, -0.2, 0.1, 0.7])
        self.assertGreater(p.vector[0], 0.2)

    def test_choleric_amplifies_negative_valence(self):
        calm, angry = VectorialPersonality("a"), VectorialPersonality("b", CHOLERIC)
        calm.update_from_action([-0.4, 0, 0, 0.5])
        angry.update_from_action([-0.4, 0, 0, 0.5])
        self.assertLess(angry.vector[0], calm.vector[0])

    def test_values_stay_in_range(self):
        p = VectorialPersonality("n")
        for _ in range(50):
            p.update_from_action([-0.4, 0.3, 0.8, 0.9])
        self.assertGreaterEqual(p.vector[0], -1.0)
        self.assertLessEqual(p.vector[2], 1.0)

    def test_npcs_have_independent_state(self):
        e = make_engine()
        e.process_action("NPC_1", "attack the NPC")
        self.assertEqual(e.personality_for("NPC_0").vector[2], 0.0)
        self.assertGreater(e.personality_for("NPC_1").vector[2], 0.0)


class MemoryTests(unittest.TestCase):
    def test_threshold_and_force(self):
        m = MemorySet()
        self.assertIsNone(m.add_importance_entry("n", [0.05, 0, 0, 0.5], {}))
        self.assertIsNotNone(m.add_importance_entry("n", [0.05, 0, 0, 0.9], {}, force=True))

    def test_consolidation_creates_unique_concepts(self):
        m = MemorySet(medium_term_retention=10)
        for _ in range(2):
            m.add_importance_entry("n", [0.5, 0, 0, 0.5], {"description": "x"})
        created = m.consolidate_and_forget(now=m.medium_term_memory[0]["timestamp"] + 11)
        self.assertEqual(len(created), 2)
        self.assertEqual(len({c["concept_id"] for c in created}), 2)
        self.assertEqual(m.medium_term_memory, [])


class ContextTests(unittest.TestCase):
    def test_switch_context(self):
        c = ContextFunction()
        c.register_context("a", {"description": "A"})
        self.assertTrue(c.switch_context("a"))
        self.assertFalse(c.switch_context("missing"))

    def test_trigger_scripts_run(self):
        c, calls = ContextFunction(), []
        c.register_trigger_script("buy", lambda npc, act: calls.append((npc, act)))
        self.assertEqual(c.execute_context_function("N", "buy a potion"), ["buy"])
        self.assertEqual(calls, [("N", "buy a potion")])
        self.assertTrue(c.should_trigger_script("buy a potion"))  # no variables needed

    def test_payload_includes_world_and_is_capped(self):
        c = ContextFunction()
        c.set_world_state({"location": "Tavern"})
        payload = c.build_strict_payload("N", [str(i) for i in range(100)], facts=["f"] * 100)
        self.assertEqual(payload[0], "World: Tavern")
        self.assertLessEqual(len(payload), 50)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.g = GraphDatabase()
        self.g.add_node("a", "NPC", name="A")
        self.g.add_node("b", "Location", name="B")

    def test_edges_are_idempotent(self):
        self.g.add_edge("a", "LOCATED_AT", "b")
        self.g.add_edge("a", "LOCATED_AT", "b", since=1)
        self.assertEqual(len(self.g.edges["a"]), 1)

    def test_subgraph_follows_incoming_edges(self):
        self.g.add_edge("a", "LOCATED_AT", "b")
        sub = self.g.subgraph_for("b", depth=1)
        self.assertEqual({n["id"] for n in sub["nodes"]}, {"a", "b"})

    def test_remove_node_drops_edges(self):
        self.g.add_edge("a", "LOCATED_AT", "b")
        self.g.remove_node("b")
        self.assertEqual(self.g.subgraph_for("a")["edges"], [])

    def test_cypher_escapes_quotes_and_keeps_edge_props(self):
        self.g.add_node("c", "Concept", name='say "hi" it\'s')
        self.g.add_edge("a", "KNOWS", "c", since="Act I")
        stmts = "\n".join(self.g.export_cypher_batch())
        self.assertIn('\\"hi\\"', stmts)
        self.assertIn("`since`: \"Act I\"", stmts)


class LLMLayerTests(unittest.TestCase):
    def test_action_is_narrated_in_second_person(self):
        self.assertEqual(narrate_action("attack the NPC"), "The player attacks you.")
        self.assertIn("dance", narrate_action("dance"))

    def test_messages_have_few_shot_and_real_situation_last(self):
        msgs = build_messages(["fact A"], "help the NPC")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn("fact A", msgs[-1]["content"])
        self.assertIn("offers to help you", msgs[-1]["content"])
        self.assertGreater(len([m for m in msgs if m["role"] == "assistant"]), 0)

    def test_clean_output_strips_think_and_prefix(self):
        self.assertEqual(clean_output("<think>x</think> Alan: \"Hi there\""), "Hi there")


class EngineTests(unittest.TestCase):
    def test_pipeline_links_events_and_moods(self):
        e = make_engine()
        r = e.process_action("NPC_0", "attack the NPC")
        self.assertIsNotNone(r["memory_event"])
        self.assertIn(r["memory_event"], e.graph.nodes)
        self.assertEqual(e.graph.get_node("NPC_0").properties["current_mood"], r["mood"])

    def test_expired_events_become_graph_memories(self):
        e = make_engine()
        e.memory.medium_term_retention = 0
        r = e.process_action("NPC_0", "attack the NPC")
        self.assertNotIn(r["memory_event"], e.graph.nodes)
        self.assertTrue(e.graph.get_nodes_by_label("LongTermMemory"))
        self.assertTrue(e.graph.get_neighbors("NPC_0", "REMEMBERED"))

    def test_llm_failure_falls_back(self):
        class Boom:
            def generate(self, payload, action):
                raise RuntimeError("down")
        e = AffektEngine(llm=Boom())
        e.initialize_modules()
        e.running = True
        r = e.process_action("NPC_1", "buy a potion")
        self.assertTrue(r["llm_used"])
        self.assertTrue(r["response"])

    def test_trigger_script_fires_from_engine(self):
        e, calls = make_engine(), []
        e.context.register_trigger_script("buy", lambda n, a: calls.append(n))
        e.process_action("NPC_1", "buy a potion")
        self.assertEqual(calls, ["NPC_1"])


if __name__ == "__main__":
    unittest.main()
