"""Regression tests for bugs found in the core modules by an adversarial audit."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.ContextFunction import ContextFunction  # noqa: E402
from modules.GraphDatabase import GraphDatabase  # noqa: E402
from modules.MemorySet import MemorySet  # noqa: E402


class MemoryTests(unittest.TestCase):
    def test_one_npcs_chatter_does_not_push_out_another_npcs_memory(self):
        m = MemorySet()
        for i in range(5):
            m.add_short_term_entry("kael", f"kael-{i}")
        for i in range(50):
            m.add_short_term_entry("marta", f"marta-{i}")
        self.assertEqual([e["phrase"] for e in m.get_short_term_context("kael")], [f"kael-{i}" for i in range(5)])

    def test_each_npc_keeps_only_its_most_recent_entries(self):
        m = MemorySet(short_term_capacity=3)
        for i in range(5):
            m.add_short_term_entry("a", str(i))
        self.assertEqual([e["phrase"] for e in m.get_short_term_context("a")], ["2", "3", "4"])

    def test_unknown_npc_has_no_memory(self):
        self.assertEqual(MemorySet().get_short_term_context("nobody"), [])

    def test_importance_of_an_empty_vector_is_zero(self):
        self.assertEqual(MemorySet.importance_of([]), 0.0)


class TriggerTests(unittest.TestCase):
    def setUp(self):
        self.c = ContextFunction()

    def test_whole_words_fire(self):
        for text in ("Please give me the key", "I want to buy it", "Sell me one", "Unlock the door", "trade you",
                     "She is selling apples", "He gave it away", "I bought a sword", "It was sold"):
            self.assertTrue(self.c.should_trigger_script(text), text)

    def test_words_that_merely_contain_a_keyword_do_not_fire(self):
        for text in ("I forgive you", "He is a buyer", "a counsellor", "that is a tirade", "bestseller"):
            self.assertFalse(self.c.should_trigger_script(text), text)

    def test_scripts_run_only_on_whole_words(self):
        fired = []
        self.c.register_trigger_script("give", lambda npc, action: fired.append(action))
        self.c.execute_context_function("NPC_1", "I forgive you")
        self.assertEqual(fired, [])
        self.c.execute_context_function("NPC_1", "give me the key")
        self.assertEqual(fired, ["give me the key"])

    def test_trigger_names_are_the_base_keyword(self):
        self.assertEqual(self.c.matched_triggers("He is buying and selling"), ["sell", "buy"])


class _Session:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.driver.queries.append((query, params))
        return self.driver.rows.get(query.split(" RETURN")[0][:12], []) if self.driver.rows else []


class _Driver:
    def __init__(self, rows=None):
        self.queries, self.rows, self.closed = [], rows, False

    def session(self):
        return _Session(self)

    def close(self):
        self.closed = True


class GraphTests(unittest.TestCase):
    def test_forgotten_nodes_are_deleted_from_neo4j_once(self):
        g = GraphDatabase()
        g.add_node("a", "NPC", name="A")
        g.add_node("e", "MediumTermEvent", description="x")
        g._driver = _Driver()
        g.sync_to_neo4j()
        g.remove_node("e")
        g.sync_to_neo4j()
        deletes = [p for q, p in g._driver.queries if "DETACH DELETE" in q]
        self.assertEqual(deletes, [{"id": "e"}])
        g.sync_to_neo4j()  # nothing new to forget
        self.assertEqual(len([1 for q, _ in g._driver.queries if "DETACH DELETE" in q]), 1)

    def test_a_node_removed_and_added_back_is_not_deleted(self):
        g = GraphDatabase()
        g.add_node("a", "NPC", name="A")
        g.remove_node("a")
        g.add_node("a", "NPC", name="A again")
        g._driver = _Driver()
        g.sync_to_neo4j()
        self.assertFalse([q for q, _ in g._driver.queries if "DETACH DELETE" in q])

    def test_removing_an_unknown_node_does_not_schedule_a_delete(self):
        g = GraphDatabase()
        g.remove_node("ghost")
        self.assertEqual(g._removed, set())

    def test_cypher_export_never_writes_nan_or_infinity(self):
        g = GraphDatabase()
        g.add_node("n", "NPC", a=float("nan"), b=float("inf"), c=1.5)
        stmt = g.export_cypher_batch()[0]
        self.assertNotIn("nan", stmt.lower())
        self.assertNotIn("inf", stmt.lower())
        self.assertIn("`a`: null", stmt)
        self.assertIn("`c`: 1.5", stmt)

    def test_load_skips_unknown_labels_and_dangling_edges_instead_of_aborting(self):
        class Rows(_Driver):
            pass

        driver = Rows()

        def run(query, **params):
            if "RETURN n.id" in query:
                return [{"id": "a", "labels": ["NPC"], "props": {"id": "a", "name": "A"}},
                        {"id": "x", "labels": ["Mystery"], "props": {"id": "x"}}]
            if "RETURN a.id" in query:
                return [{"src": "a", "rel": "KNOWS", "dst": "x", "props": {}},
                        {"src": "a", "rel": "HATES", "dst": "a", "props": {}}]
            return []

        class Session(_Session):
            def run(self, query, **params):
                return run(query, **params)

        driver.session = lambda: Session(driver)
        g = GraphDatabase()
        g._driver = driver
        self.assertTrue(g.load_from_neo4j())
        self.assertEqual(list(g.nodes), ["a"])
        self.assertEqual(dict(g.edges), {})


if __name__ == "__main__":
    unittest.main()
