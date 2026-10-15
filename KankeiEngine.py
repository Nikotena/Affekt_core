#!/usr/bin/env python3
"""
Kankei Engine - Main Engine Controller
Coordinates all modules and implements the 80/20 architecture
"""

import time
from typing import Dict, List, Optional

from modules.VectorialPersonality import VectorialPersonality
from modules.MemorySet import MemorySet
from modules.ContextFunction import ContextFunction
from modules.GraphDatabase import GraphDatabase
from modules.IntentParser import IntentParser, is_commerce
from modules.LLMLayer import LLMBackend, StubBackend, FALLBACK_LINES, style_for

class KankeiEngine:
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 connect_neo4j: bool = False,
                 llm: Optional[LLMBackend] = None):
        # Core modules (80/20 architecture: 80% deterministic, 20% AI)
        self.llm: LLMBackend = llm or StubBackend()
        self.personalities: Dict[str, VectorialPersonality] = {}
        self.parser = IntentParser()
        self.memory = MemorySet()
        self.context = ContextFunction()
        # Passive persistence layer (in-memory primary graph + optional Neo4j)
        self.graph = GraphDatabase(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            auto_connect=connect_neo4j,
        )
        
        # Engine state
        self.running = False
        self.update_interval = 0.1  # seconds
        self._event_counter = 0
        self._memory_counter = 0
        
        # Register initial contexts (stored in the ContextFunction registry)
        self._register_initial_contexts()
        
        # Seed the world graph
        self._seed_world_graph()
    
    def _register_initial_contexts(self):
        """Register initial context definitions"""
        # Chapter-based contexts
        self.context.contexts["chapter_start"] = {
            "description": "Initial chapter setup",
            "variables": ["chapter", "location", "goal"]
        }
        
        self.context.contexts["shop_open"] = {
            "description": "Shop opened",
            "variables": ["shop_items", "prices"]
        }
        
        self.context.contexts["npc_interaction"] = {
            "description": "Interacting with NPCs",
            "variables": ["npc_id", "intent"]
        }
    
    def initialize_modules(self):
        """Initialize all modules"""
        self.personality_for("NPC_0").set_temperament(0)  # Normal temperament
        self.context.switch_context("chapter_start")

        # Initial context setup
        self.context.set_world_state({
            "location": "Tavern",
            "chapter": "Act I",
            "time_of_day": "evening"
        })
        
        print("Kankei Engine initialized successfully")
    
    @property
    def personality(self) -> VectorialPersonality:
        """Default NPC's personality (kept for backwards compatibility)."""
        return self.personality_for("NPC_0")

    def personality_for(self, npc_id: str) -> VectorialPersonality:
        """Each NPC has its own emotional state, created on first use."""
        if npc_id not in self.personalities:
            self.personalities[npc_id] = VectorialPersonality(npc_id)
        return self.personalities[npc_id]

    def _graph_facts(self, npc_id: str, subgraph: Dict, action: str) -> List[str]:
        """Natural-language facts from the NPC's neighbourhood, filtered by relevance to the action."""
        nodes = {n["id"]: n for n in subgraph["nodes"]}
        commerce = is_commerce(action)
        facts = []
        for e in subgraph["edges"]:
            if e["source"] != npc_id:
                continue
            target = nodes.get(e["target"])
            if target is None:
                continue
            name, props = target["properties"].get("name"), target["properties"]
            if e["type"] == "LOCATED_AT":
                facts.append(f"You are at the {name}.")
            elif e["type"] == "KNOWS":
                facts.append(f"You know the {name.lower()}" + (f" since {e['properties']['since']}." if e["properties"].get("since") else "."))
            elif e["type"] == "OWNS" and commerce:
                facts.append(f"You sell a {name} for {props.get('price', '?')} coins.")
        return facts

    def _consolidate(self) -> None:
        """Run memory consolidation and mirror new long-term concepts into the graph."""
        for concept in self.memory.consolidate_and_forget():
            self._memory_counter += 1
            node_id = f"memory_{concept['npc_id']}_{self._memory_counter}"
            info = concept["data"]["key_info"]
            self.graph.add_node(node_id, "LongTermMemory", concept=info.get("description", ""),
                                strength=concept["data"]["original_importance"])
            if self.graph.get_node(concept["npc_id"]) is not None:
                self.graph.add_edge(concept["npc_id"], "REMEMBERED", node_id)
            # The raw event is forgotten; only the consolidated memory stays in the graph.
            if info.get("node_id"):
                self.graph.remove_node(info["node_id"])

    def _seed_world_graph(self):
        """Populate the in-memory graph with the initial world topology."""
        # Player
        self.graph.add_node("player_01", "Player", name="Player", location="tavern_01")
        # NPCs
        self.graph.add_node("NPC_0", "NPC", name="Alan", role="Bartender", current_mood="neutral")
        self.graph.add_node("NPC_1", "NPC", name="Marta", role="Merchant", current_mood="neutral")
        self.graph.add_node("NPC_2", "NPC", name="Guard", role="Guard", current_mood="neutral")
        self.graph.add_node("NPC_3", "NPC", name="Elena", role="Healer", current_mood="neutral")
        self.graph.add_node("NPC_4", "NPC", name="Kael", role="Blacksmith", current_mood="neutral")
        # Locations
        self.graph.add_node("tavern_01", "Location", name="Tavern", region="Mountain Pass")
        self.graph.add_node("market_01", "Location", name="Market", region="Mountain Pass")
        # Items
        self.graph.add_node("item_potion", "Item", name="Potion", price=25, rarity="common")
        self.graph.add_node("item_sword", "Item", name="Sword", price=100, rarity="uncommon")
        # Concepts
        self.graph.add_node("concept_debt", "Concept", name="Debt", category="social")
        
        # Relationships
        self.graph.add_edge("player_01", "LOCATED_AT", "tavern_01")
        self.graph.add_edge("NPC_0", "LOCATED_AT", "tavern_01")
        self.graph.add_edge("NPC_1", "LOCATED_AT", "market_01")
        self.graph.add_edge("NPC_2", "LOCATED_AT", "tavern_01")
        self.graph.add_edge("NPC_0", "KNOWS", "player_01", since="Act I")
        self.graph.add_edge("NPC_1", "OWNS", "item_potion")
        self.graph.add_edge("NPC_2", "OWNS", "item_sword")
    
    def _generate_action_delta(self, action: str) -> List[float]:
        """Emotional push of a player action, from the rule-based intent parser."""
        return list(self.parser.parse(action).delta)

    def process_action(self, npc_id: str, action: str) -> Dict:
        """Process a player action and return the engine response"""
        if not self.running:
            raise RuntimeError("Engine is not running")
        
        # Step 1: Vectorial Personality calculation (per NPC)
        personality = self.personality_for(npc_id)
        parsed = self.parser.parse(action, names=self.names_for(npc_id))
        text = parsed.clean or action  # what the player said without addressing the NPC by name
        action_delta = list(parsed.delta)
        personality.update_from_action(action_delta)
        npc_node = self.graph.get_node(npc_id)
        if npc_node is not None:
            npc_node.properties["current_mood"] = personality.mood()

        # Step 2: Memory write orchestration
        self.memory.add_short_term_entry(npc_id, action)
        importance = self.memory.importance_of(action_delta)
        fast_track = action_delta[3] >= 0.8  # high temporal inertia is always remembered
        event_node_id = None
        fact = self.memory.add_importance_entry(
            npc_id, action_delta, {"description": action}, force=fast_track)
        if fact is not None:
            self._event_counter += 1
            event_node_id = f"event_{npc_id}_{self._event_counter}"
            fact["data"]["node_id"] = event_node_id
            self.graph.add_node(
                event_node_id,
                "MediumTermEvent",
                description=action,
                timestamp_created=fact["timestamp"],
                importance=importance,
            )
            # Link NPC -> EXPERIENCED -> event, Player -> CAUSED -> event
            if npc_node is not None:
                self.graph.add_edge(npc_id, "EXPERIENCED", event_node_id)
            if self.graph.get_node("player_01") is not None:
                self.graph.add_edge("player_01", "CAUSED", event_node_id)

        # Step 3: Context function evaluation (runs registered trigger scripts)
        triggered = self.context.execute_context_function(npc_id, text)

        # Step 4: Build strict payload for LLM
        payload, subgraph = self.build_payload(npc_id, text)
        # The LLM is only called for meaningful moments; everything else is deterministic.
        use_llm = (self.context.should_trigger_script(text) or event_node_id is not None
                   or parsed.intent != "other" or len(text.split()) > 2)
        response = self._generate_response(payload, npc_id, text, use_llm, personality.mood())

        # Step 5: Consolidate and forget
        self._consolidate()

        return {
            "npc_id": npc_id,
            "action": action,
            "response": response,
            "mood": personality.mood(),
            "intent": parsed.intent,
            "offer": parsed.offer,
            "addressed": parsed.addressed,
            "triggered": triggered,
            "llm_used": use_llm,
            "payload": payload,
            "payload_size": len(payload),
            "graph_nodes": len(subgraph["nodes"]),
            "graph_edges": len(subgraph["edges"]),
            "memory_event": event_node_id,
            "timestamp": time.time(),
        }

    def names_for(self, npc_id: str) -> List[str]:
        """Names the player may use to address this NPC: its name plus any nicknames."""
        node = self.graph.get_node(npc_id)
        if node is None:
            return []
        return [n for n in [node.properties.get("name"), *node.properties.get("aliases", [])] if n]

    def add_alias(self, npc_id: str, alias: str) -> None:
        """Register a nickname the player uses for an NPC ("Big Al", "old man")."""
        node = self.graph.get_node(npc_id)
        if node is None:
            raise KeyError(npc_id)
        aliases = node.properties.setdefault("aliases", [])
        if alias and alias.lower() not in (a.lower() for a in aliases):
            aliases.append(alias)

    def build_payload(self, npc_id: str, action: str):
        """Assemble the LLM payload for an NPC from its current state. Returns (payload, subgraph)."""
        personality = self.personality_for(npc_id)
        npc_node = self.graph.get_node(npc_id)
        subgraph = self.graph.subgraph_for(npc_id, depth=2)
        facts = []
        if npc_node is not None:
            facts.append(f"You are {npc_node.properties['name']}, the {npc_node.properties['role']}.")
        facts += self._graph_facts(npc_id, subgraph, action)
        facts.append(style_for(personality.mood(), personality.vector[1]))
        recent = [f"Player earlier: {e['phrase']}" for e in self.memory.get_short_term_context(npc_id)]
        return self.context.build_strict_payload(npc_id, recent, facts=facts), subgraph

    def _generate_response(self, payload: List[str], npc_id: str, action: str,
                           use_llm: bool, mood: str) -> str:
        """LLM phrasing for important moments; deterministic line otherwise or on failure."""
        if use_llm:
            try:
                text = self.llm.generate(payload, action)
                if text:
                    return text
            except Exception as e:
                print(f"[Engine] LLM failed for {npc_id}: {e}")
        return FALLBACK_LINES.get(mood, FALLBACK_LINES["neutral"])

    def run(self, num_actions: int) -> List[Dict]:
        """Run the engine for a given number of actions"""
        results = []
        sample_actions = [
            "greet the NPC",
            "ask about the quest",
            "buy a potion from me",
            "attack the NPC",
            "help the NPC",
        ]
        
        for i in range(num_actions):
            # Rotate through different NPCs
            npc_id = f"NPC_{i % 5}"  # NPC_0, NPC_1, ...
            
            # Rotate through sample actions
            action = sample_actions[i % len(sample_actions)]
            
            result = self.process_action(npc_id, action)
            results.append(result)
        
        return results

    def shutdown(self):
        """Gracefully shut down: sync graph to Neo4j (if connected)."""
        self.running = False
        self._consolidate()
        self.graph.close()

    def save_snapshot(self, path: str) -> None:
        """Write the graph as a batch of Cypher statements (offline analysis)."""
        with open(path, "w", encoding="utf-8") as f:
            for stmt in self.graph.export_cypher_batch():
                f.write(stmt + "\n")


def main():
    """Main entry point"""
    engine = KankeiEngine()
    engine.initialize_modules()
    engine.running = True
    
    # Run for 20 actions
    results = engine.run(20)
    
    print("\n=== Engine Results ===")
    for i, result in enumerate(results):
        print(f"Action {i+1}: {result['npc_id']} - '{result['action']}'")
        print(f"  Payload size: {result['payload_size']} tokens")
        print(f"  Graph subgraph: {result['graph_nodes']} nodes, {result['graph_edges']} edges")
        if result["memory_event"]:
            print(f"  Memory event persisted: {result['memory_event']}")
    
    print(f"\nIn-memory graph: {len(engine.graph.nodes)} nodes")
    print(f"Neo4j persistence active: {engine.graph.is_persistent()}")
    
    engine.shutdown()
    print("\nEngine shutdown complete.")


if __name__ == "__main__":
    main()
