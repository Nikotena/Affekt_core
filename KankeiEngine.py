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

class KankeiEngine:
    def __init__(self):
        # Core modules (80/20 architecture: 80% deterministic, 20% AI)
        self.personality = VectorialPersonality("NPC_0")
        self.memory = MemorySet()
        self.context = ContextFunction()
        
        # Engine state
        self.running = False
        self.last_update = 0
        self.update_interval = 0.1  # seconds
        
        # Context registry
        self.contexts = {}
        
        # Register initial contexts
        self._register_initial_contexts()
    
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
        
        # Step 2: Memory lookup
        short_term = self.memory.get_short_term_context(npc_id)
        medium_term = self.memory.get_medium_term_context(npc_id)
        
        # Convert short-term entries to strings for context function
        short_term_strings = [f"{entry['npc_id']}: {entry['phrase']}" for entry in short_term]
        
        # Step 3: Context function evaluation
        triggers_script = self.context.should_trigger_script(action, short_term_strings)
        
        if triggers_script:
            self.context.execute_context_function(npc_id, action)
        
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
        
        for i in range(num_actions):
            # Rotate through different NPCs
            npc_id = f"NPC_{i % 5}"  # NPC_0, NPC_1, ...
            
            # Random action
            action = f"interact with {npc_id}"
            
            result = self.process_action(npc_id, action)
            results.append(result)
            
            # Periodic consolidation
            if i % 10 == 0:
                self.memory.consolidate_and_forget()
        
        return results


def main():
    """Main entry point"""
    engine = KankeiEngine()
    engine.initialize_modules()
    
    # Run for 20 actions
    results = engine.run(20)
    
    print("\n=== Engine Results ===")
    for i, result in enumerate(results):
        print(f"Action {i+1}: {result['npc_id']} - Response generated")
        print(f"  Payload size: {result['payload_size']} tokens")
    
    print("\nEngine shutdown complete.")


if __name__ == "__main__":
    main()
