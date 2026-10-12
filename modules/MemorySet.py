"""
Memory Set (Memory Engine)
Implements the 3-layer memory hierarchy: Short-Term (RAM), Medium-Term, Long-Term
"""

import time
from typing import Dict, List, Optional, Any
from collections import deque

class MemorySet:
    def __init__(self, short_term_capacity: int = 20, medium_term_retention: int = 300):  # 5 minutes in seconds
        self.short_term_buffer: deque = deque(maxlen=short_term_capacity)
        self.medium_term_memory: List[Dict] = []
        self.long_term_memory: List[Dict] = []
        
        # Constants
        self.medium_term_retention = medium_term_retention  # seconds (default 5 minutes)
        self.importance_threshold = 0.15  # Δ ≥ 0.15 triggers medium-term fact creation
        
        # Memory structure:
        # Each entry has: id, type, data, timestamp, importance
        self.memory_counter = 0
    
    def add_short_term_entry(self, npc_id: str, phrase: str, metadata: Optional[Dict] = None) -> None:
        """Add a phrase to the short-term ring buffer"""
        entry = {
            "npc_id": npc_id,
            "phrase": phrase,
            "timestamp": time.time(),
            "type": "short_term"
        }
        if metadata:
            entry.update(metadata)
        self.short_term_buffer.append(entry)
    
    def add_importance_entry(self, npc_id: str, delta_vector: List[float], 
                            event_data: Dict) -> Optional[Dict]:
        """
        Evaluate if interaction exceeds importance threshold.
        If Δ ≥ 0.15 on any axis, creates medium-term fact.
        """
        # Check if vector delta exceeds threshold
        importance = max(abs(delta) for delta in delta_vector[:3])
        
        # If exceeds threshold, create medium-term fact
        if importance >= self.importance_threshold:
            fact_entry = {
                "npc_id": npc_id,
                "fact_type": "memory",
                "data": event_data,
                "timestamp": time.time(),
                "importance": importance
            }
            self.medium_term_memory.append(fact_entry)
            return fact_entry
        
        # Check temporal inertia - if W >= 0.8, fast-track to medium-term
        # (This would be accessed from personality module)
        return None
    
    def add_long_term_concept(self, npc_id: str, concept_id: str, 
                              concept_data: Dict, consolidation_time: float) -> None:
        """Add a long-term concept after consolidation"""
        entry = {
            "npc_id": npc_id,
            "concept_id": concept_id,
            "data": concept_data,
            "consolidated_at": time.time() - consolidation_time,
            "type": "long_term"
        }
        self.long_term_memory.append(entry)
    
    def get_short_term_context(self, npc_id: str) -> List[Dict]:
        """Retrieve short-term entries for a specific NPC"""
        return [entry for entry in self.short_term_buffer 
                if entry["npc_id"] == npc_id]
    
    def get_medium_term_context(self, npc_id: str) -> List[Dict]:
        """Retrieve medium-term memories for a specific NPC"""
        return [entry for entry in self.medium_term_memory 
                if entry["npc_id"] == npc_id]
    
    def get_long_term_context(self, npc_id: str) -> List[Dict]:
        """Retrieve long-term concepts for a specific NPC"""
        return [entry for entry in self.long_term_memory 
                if entry["npc_id"] == npc_id]
    
    def consolidate_and_forget(self) -> None:
        """Background consolidation and forgetting algorithm"""
        current_time = time.time()
        
        # Process medium-term memories
        new_medium_term = []
        for entry in self.medium_term_memory:
            age_seconds = current_time - entry["timestamp"]
            
            if age_seconds < self.medium_term_retention:
                # Still within retention window
                new_medium_term.append(entry)
            else:
                # Age past retention - convert to long-term concept
                # Extract key facts and create abstraction
                concept_id = f"CONCEPT_{entry['npc_id']}_{entry.get('fact_type', 'unknown')}"
                concept_data = {
                    "key_info": entry["data"],
                    "original_importance": entry["importance"],
                    "source_timestamp": entry["timestamp"]
                }
                self.add_long_term_concept(
                    entry["npc_id"], 
                    concept_id, 
                    concept_data, 
                    age_seconds
                )
        
        self.medium_term_memory = new_medium_term