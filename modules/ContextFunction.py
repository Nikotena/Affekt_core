"""
Context Function Module
Scene orchestrator that determines triggers and assembles strict payloads for LLM
"""

import re
from typing import Callable, Dict, List, Optional

# trigger name -> the whole words that fire it. Whole words only: "forgive" must not fire "give",
# nor "counsellor" fire "sell".
TRIGGER_WORDS = {
    "sell": ("sell", "sells", "selling", "sold"),
    "give": ("give", "gives", "giving", "gave"),
    "unlock": ("unlock", "unlocks", "unlocking", "unlocked"),
    "buy": ("buy", "buys", "buying", "bought"),
    "trade": ("trade", "trades", "trading", "traded"),
}
TRIGGER_KEYWORDS = tuple(TRIGGER_WORDS)
TRIGGER_NPC_IDS = ("bartender_01", "merchant_01")
_TRIGGER_RES = [(name, re.compile(r"\b(?:" + "|".join(words) + r")\b")) for name, words in TRIGGER_WORDS.items()]
_TRIGGER_RES += [(npc, re.compile(r"\b" + re.escape(npc) + r"\b")) for npc in TRIGGER_NPC_IDS]

MAX_PAYLOAD_LINES = 50
MAX_CONTEXT_VARIABLES = 10


class ContextFunction:
    def __init__(self):
        self.world_state: Dict = {}
        self.contexts: Dict[str, Dict] = {}
        self.active_context_id: Optional[str] = None
        self.active_context: Optional[Dict] = None
        self.trigger_scripts: Dict[str, Callable[[str, str], None]] = {}

    def set_world_state(self, world_state: Dict) -> None:
        """Set the current world state (chapter, location, etc.)"""
        self.world_state = world_state

    def register_context(self, context_id: str, context_data: Dict) -> None:
        self.contexts[context_id] = context_data

    def switch_context(self, new_context_id: str) -> bool:
        """Swap the active context. Returns False if it is not registered."""
        if new_context_id not in self.contexts:
            return False
        self.active_context_id = new_context_id
        self.active_context = self.contexts[new_context_id]
        return True

    def get_active_context(self) -> Optional[Dict]:
        return self.active_context

    def register_trigger_script(self, trigger_name: str, callback: Callable[[str, str], None]) -> None:
        """Register a callback(npc_id, action) run when the keyword/NPC `trigger_name` fires."""
        self.trigger_scripts[trigger_name] = callback

    def matched_triggers(self, event_description: str) -> List[str]:
        """Names of all triggers (keywords / NPC ids) present in the event."""
        text = event_description.lower()
        return [name for name, pattern in _TRIGGER_RES if pattern.search(text)]

    def should_trigger_script(self, event_description: str,
                              context_variables: Optional[List[str]] = None) -> bool:
        """Rule-based trigger check (keywords or well-known NPC ids)."""
        return bool(self.matched_triggers(event_description))

    def execute_context_function(self, npc_id: str, action: str) -> List[str]:
        """Run the registered scripts for every trigger the action fires; returns their names."""
        fired = []
        for name in self.matched_triggers(action):
            callback = self.trigger_scripts.get(name)
            if callback is not None:
                callback(npc_id, action)
                fired.append(name)
        return fired

    def build_strict_payload(self, npc_id: str, context_variables: List[str],
                             world_state: Optional[Dict] = None,
                             facts: Optional[List[str]] = None) -> List[str]:
        """Assemble the compact, ordered line list sent to the LLM.

        Order is priority order, so truncation drops the least important lines:
        world, NPC, active scene, graph facts, then the most recent context variables.
        """
        world = self.world_state if world_state is None else world_state
        payload: List[str] = []
        if world:
            payload.append(f"World: {world.get('location', 'unknown')}")
            if world.get("part_of_day"):  # set by the host from the player's clock (engine.set_time_of_day)
                payload.append(f"Time: {world['part_of_day']}.")
        payload.append(f"NPC: {npc_id}")
        if self.active_context:
            payload.append(f"Scene: {self.active_context.get('description', self.active_context_id)}")
        payload.extend(facts or [])
        if context_variables:
            payload.extend(context_variables[-MAX_CONTEXT_VARIABLES:])
        return payload[:MAX_PAYLOAD_LINES]
