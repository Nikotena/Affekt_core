"""
Memory Set (Memory Engine)
Implements the 3-layer memory hierarchy: Short-Term (RAM), Medium-Term, Long-Term
"""

import time
from collections import deque
from typing import Dict, List, Optional


class MemorySet:
    def __init__(self, short_term_capacity: int = 20, medium_term_retention: float = 300):
        self.short_term_buffer: deque = deque(maxlen=short_term_capacity)
        self.medium_term_memory: List[Dict] = []
        self.long_term_memory: List[Dict] = []

        self.medium_term_retention = medium_term_retention  # seconds (default 5 minutes)
        self.importance_threshold = 0.15  # delta >= 0.15 on any axis creates a medium-term fact
        self._concept_counter = 0

    @staticmethod
    def importance_of(delta_vector: List[float]) -> float:
        """Largest absolute change on the X/Y/Z axes."""
        return max(abs(d) for d in delta_vector[:3])

    def add_short_term_entry(self, npc_id: str, phrase: str, metadata: Optional[Dict] = None) -> None:
        """Add a phrase to the short-term ring buffer"""
        entry = {"npc_id": npc_id, "phrase": phrase, "timestamp": time.time(), "type": "short_term"}
        if metadata:
            entry.update(metadata)
        self.short_term_buffer.append(entry)

    def add_importance_entry(self, npc_id: str, delta_vector: List[float],
                             event_data: Dict, force: bool = False) -> Optional[Dict]:
        """Create a medium-term fact if the interaction is important enough (or `force`d,
        e.g. high temporal inertia fast-tracks the event)."""
        importance = self.importance_of(delta_vector)
        if not force and importance < self.importance_threshold:
            return None
        fact = {
            "npc_id": npc_id,
            "fact_type": "memory",
            "data": event_data,
            "timestamp": time.time(),
            "importance": importance,
        }
        self.medium_term_memory.append(fact)
        return fact

    def add_long_term_concept(self, npc_id: str, concept_id: str, concept_data: Dict,
                              consolidated_at: Optional[float] = None) -> Dict:
        """Store a consolidated concept and return it."""
        entry = {
            "npc_id": npc_id,
            "concept_id": concept_id,
            "data": concept_data,
            "consolidated_at": consolidated_at if consolidated_at is not None else time.time(),
            "type": "long_term",
        }
        self.long_term_memory.append(entry)
        return entry

    def get_short_term_context(self, npc_id: str) -> List[Dict]:
        return [e for e in self.short_term_buffer if e["npc_id"] == npc_id]

    def get_medium_term_context(self, npc_id: str) -> List[Dict]:
        return [e for e in self.medium_term_memory if e["npc_id"] == npc_id]

    def get_long_term_context(self, npc_id: str) -> List[Dict]:
        return [e for e in self.long_term_memory if e["npc_id"] == npc_id]

    def consolidate_and_forget(self, now: Optional[float] = None) -> List[Dict]:
        """Promote expired medium-term facts to long-term concepts.

        Returns the newly created long-term entries so callers can mirror them
        (e.g. into the graph).
        """
        now = time.time() if now is None else now
        kept, created = [], []
        for entry in self.medium_term_memory:
            if now - entry["timestamp"] < self.medium_term_retention:
                kept.append(entry)
                continue
            self._concept_counter += 1
            concept_id = f"CONCEPT_{entry['npc_id']}_{self._concept_counter}"
            created.append(self.add_long_term_concept(
                entry["npc_id"],
                concept_id,
                {
                    "key_info": entry["data"],
                    "original_importance": entry["importance"],
                    "source_timestamp": entry["timestamp"],
                },
                consolidated_at=now,
            ))
        self.medium_term_memory = kept
        return created
