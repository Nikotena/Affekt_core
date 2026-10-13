#!/usr/bin/env python3
"""
Kankei Engine - Main Engine Controller
Coordinates all modules and implements the 80/20 architecture
"""

import sys
import time
import os
from typing import Dict, List, Optional

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from modules.VectorialPersonality import VectorialPersonality
from modules.MemorySet import MemorySet
from modules.ContextFunction import ContextFunction
from modules.GraphDatabase import GraphDatabase

class KankeiEngine:
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 connect_neo4j: bool = False):
        # Core modules (80/20 architecture: 80% deterministic, 20% AI)
        self.personality = VectorialPersonality("NPC_0")
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
        self.last_update = 0
        self.update_interval = 0.1  # seconds
        self._event_counter = 0
        
        # Context registry
        self.contexts = {}
        
        # Register initial contexts
        self._register_initial_contexts()
        
        # Seed the world graph
        self._seed_world_graph()
    
    def _register_initial_contexts(self):
        """Register initial context definitions"""
        # Chapter-based contexts
        self.contexts["chapter_start"] = {
            "description": "Initial chapter setup",
            "variables": ["chapter", "location", "goal"]
        }
        
        self.contexts["shop_open"] = {
            "description": "Shop opened",
            "variables": ["shop_items", "prices"]
        }
        
        self.contexts["npc_interaction"] = {
            "description": "Interacting with NPCs",
            "variables": ["npc_id", "intent"]
        }
    
    def initialize_modules(self):
        """Initialize all modules"""
        # Ensure personality is properly configured
        self.personality.set_temperament(0)  # Normal temperament
        
        # Initial context setup
        self.context.set_world_state({
            "location": "Tavern",
            "chapter": "Act I",
            "time_of_day": "evening"
        })
        
        print("Kankei Engine initialized successfully")
    
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
        """Generate a delta vector based on the action string"""
        # Simple heuristic for different actions
        if "greet" in action.lower():
            return [0.3, -0.2, 0.1, 0.7]  # Positive emotional impact
        elif "fight" in action.lower() or "attack" in action.lower():
            return [-0.4, 0.3, 0.8, 0.9]  # High stress, hostile
        elif "ask" in action.lower() or "question" in action.lower():
            return [0.2, 0.1, 0.3, 0.5]  # Moderate impact
        elif "help" in action.lower() or "aid" in action.lower():
            return [0.4, -0.1, 0.2, 0.6]  # Supportive action
        else:
            return [0.1, 0.1, 0.1, 0.5]  # Default small impact
    
    def process_action(self, npc_id: str, action: str) -> Dict:
        """Process a player action and return the engine response"""
        if not self.running:
            raise RuntimeError("Engine is not running")
        
        # Step 1: Vectorial Personality calculation
        action_delta = self._generate_action_delta(action)
        delta = self.personality.update_from_action(action_delta)
        
        # Step 2: Memory lookup + write orchestration
        self.memory.add_short_term_entry(npc_id, action)
        importance = max(abs(d) for d in action_delta[:3])
        event_node_id = None
        if importance >= self.memory.importance_threshold or action_delta[3] >= 0.8:
            self._event_counter += 1
            event_node_id = f"event_{npc_id}_{self._event_counter}"
            self.graph.add_node(
                event_node_id,
                "MediumTermEvent",
                description=action,
                timestamp_created=time.time(),
                importance=importance,
            )
            self.memory.add_importance_entry(npc_id, action_delta, {"description": action, "node_id": event_node_id})
            # Link NPC -> EXPERIENCED -> event, Player -> CAUSED -> event
            if self.graph.get_node(npc_id) is not None:
                self.graph.add_edge(npc_id, "EXPERIENCED", event_node_id)
            if self.graph.get_node("player_01") is not None:
                self.graph.add_edge("player_01", "CAUSED", event_node_id)
        
        short_term = self.memory.get_short_term_context(npc_id)
        medium_term = self.memory.get_medium_term_context(npc_id)
        
        # Convert short-term entries to strings for context function
        short_term_strings = [f"{entry['npc_id']}: {entry['phrase']}" for entry in short_term]
        
        # Step 3: Context function evaluation
        triggers_script = self.context.should_trigger_script(action, short_term_strings)
        
        if triggers_script:
            self.context.execute_context_function(npc_id, action)
        
        # Extract a graph subgraph around the NPC to enrich the payload
        subgraph = self.graph.subgraph_for(npc_id, depth=2)
        
        # Step 4: Build strict payload for LLM
        payload = self.context.build_strict_payload(
            npc_id,
            short_term_strings,
            self.context.get_active_context()
        )
        
        # Simulate LLM response
        response = self._generate_response(payload, npc_id, action)
        
        # Step 5: Consolidate and forget (background task)
        self.memory.consolidate_and_forget()
        
        return {
            "npc_id": npc_id,
            "action": action,
            "response": response,
            "payload_size": len(payload),
            "graph_nodes": len(subgraph["nodes"]),
            "graph_edges": len(subgraph["edges"]),
            "memory_event": event_node_id,
            "timestamps": [
                self.last_update,
                time.time()
            ]
        }
    
    def _generate_response(self, payload: List[str], npc_id: str, action: str) -> str:
        """Generate a simulated LLM response"""
        # In a real implementation, this would call the local micro-LLM
        # For now, we return a formatted response based on the payload
        
        response_lines = [
            f"Response to action: {action}",
            f"NPC: {npc_id}",
            f"Context: {len(payload)} variables considered",
            "---",
            "Narrative response generated by local micro-LLM"
        ]
        return "\n".join(response_lines)
    
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
            
            # Periodic consolidation
            if i % 10 == 0:
                self.memory.consolidate_and_forget()
        
        return results

    def shutdown(self):
        """Gracefully shut down: sync graph to Neo4j (if connected)."""
        self.running = False
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
