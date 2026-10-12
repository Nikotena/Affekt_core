"""
Context Function Module
Scene orchestrator that determines triggers and assembles strict payloads for LLM
"""

from typing import Dict, List, Optional, Callable
import json

class ContextFunction:
    def __init__(self):
        self.world_state: Dict = {}
        self.active_context: Optional[Dict] = None
        self.context_switch_callbacks: List[Callable] = []
        self.trigger_scripts: Dict[str, Callable] = {}
    
    def set_world_state(self, world_state: Dict) -> None:
        """Set the current world state (chapter, location, etc.)"""
        self.world_state = world_state
    
    def register_context(self, context_id: str, context_data: Dict) -> None:
        """Register a context definition"""
        self.contexts[context_id] = context_data
    
    def register_trigger_script(self, trigger_name: str, callback: Callable) -> None:
        """Register a script that executes when a trigger fires"""
        self.trigger_scripts[trigger_name] = callback
    
    def should_trigger_script(self, event_description: str, 
                             context_variables: List[str]) -> bool:
        """
        Determine if an event triggers a script in the game
        """
        # Simple rule-based triggering
        if not context_variables:
            return False
        
        # Check for specific keywords
        trigger_keywords = ["sell", "give", "unlock", "buy", "trade"]
        if any(keyword in event_description.lower() for keyword in trigger_keywords):
            return True
        
        # Check for specific NPC IDs
        if "bartender_01" in event_description or "merchant_01" in event_description:
            return True
        
        return False
    
    def build_strict_payload(self, npc_id: str, 
                            context_variables: List[str],
                            world_state: Optional[Dict] = None) -> List[str]:
        """
        Build a strict payload (16-128 tokens) for the LLM
        """
        payload = []
        
        # Add world state info
        if world_state:
            payload.append(f"World: {world_state.get('location', 'unknown')}")
        
        # Add NPC info
        payload.append(f"NPC: {npc_id}")
        
        # Add context variables (limited to 16-128 tokens)
        if context_variables:
            # Limit to reasonable length
            max_vars = min(10, len(context_variables))
            for i in range(max_vars):
                payload.append(context_variables[i])
        
        # Limit to reasonable length
        if len(payload) > 50:
            payload = payload[:50]
        
        return payload
    
    def execute_context_function(self, npc_id: str, action: str) -> None:
        """Execute context function (simulated)"""
        print(f"Context function executed for NPC {npc_id} with action: {action}")
        
        # In a real implementation, this would interact with the game engine
        # and possibly invoke scripts/events
        if self.should_trigger_script(action, []):
            print(f"Triggered script for action: {action}")
    
    def switch_context(self, new_context_id: str) -> bool:
        """Atomically swap the active context"""
        if new_context_id in self.contexts:
            self.active_context = self.contexts[new_context_id]
            return True
        return False
    
    def get_active_context(self) -> Optional[Dict]:
        """Get the currently active context"""
        return self.active_context