"""Save and restore one player's session: what every NPC feels and remembers.

    state = export_state(engine)        # plain JSON-able dict, no keys, no secrets
    import_state(engine, state)         # validates everything first; raises StateError and changes nothing if invalid

The file may come from anywhere (a player edits it, an attacker crafts it), so import never trusts it: fixed schema,
size caps, finite numbers clamped to their range, text trimmed and stripped of control characters, only known NPC ids.
Times are stored as ages ("how old was this memory when you saved"), so time spent away does not make it fade.
"""
import math
import time
from collections import deque
from typing import Any, Dict, List

from modules.VectorialPersonality import _RANGES

FORMAT = "affekt-session"
VERSION = 1
MAX_BYTES = 200_000           # size of the file the server accepts
MAX_TEXT = 300                # characters kept per phrase
MAX_SHORT, MAX_MEDIUM, MAX_LONG = 20, 50, 100   # entries kept per NPC
MAX_AGE = 10_000_000.0        # seconds


class StateError(ValueError):
    """The session file is not valid. Nothing was changed."""


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise StateError(f"{what}: expected text")
    return "".join(c for c in value if c >= " " and c != "\x7f")[:MAX_TEXT].strip()


def _num(value: Any, lo: float, hi: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StateError(f"{what}: expected a finite number")
    return min(hi, max(lo, float(value)))


def _list(value: Any, cap: int, what: str) -> list:
    if not isinstance(value, list):
        raise StateError(f"{what}: expected a list")
    return value[-cap:]  # if too long, keep the most recent


def _npc_ids(engine) -> List[str]:
    return sorted(n.node_id for n in engine.graph.get_nodes_by_label("NPC"))


# ------------------------------------------------------------------ export
def export_state(engine) -> Dict[str, Any]:
    now = time.time()
    npcs: Dict[str, Any] = {}
    for npc_id in _npc_ids(engine):
        p = engine.personalities.get(npc_id)
        long_term = engine.memory.get_long_term_context(npc_id)
        npcs[npc_id] = {
            "vector": [round(v, 4) for v in (p.vector if p else (0.0, 0.0, 0.0, 0.5))],
            "last_reply": engine._last_reply.get(npc_id) or "",
            "short_term": [{"phrase": e["phrase"], "age_s": round(max(0.0, now - e["timestamp"]), 2)}
                           for e in engine.memory.get_short_term_context(npc_id)],
            "medium_term": [{"description": e["data"].get("description", ""), "importance": round(e["importance"], 4),
                             "age_s": round(max(0.0, now - e["timestamp"]), 2)}
                            for e in engine.memory.get_medium_term_context(npc_id)],
            "long_term": [{"description": (e["data"].get("key_info") or {}).get("description", ""),
                           "importance": round(e["data"].get("original_importance", 0.0), 4),
                           "age_s": round(max(0.0, now - e.get("consolidated_at", now)), 2)}
                          for e in long_term],
        }
    return {
        "format": FORMAT,
        "version": VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "npcs": npcs,
    }


# ------------------------------------------------------------------ validate (pure: builds a clean copy)
def validate_state(engine, data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise StateError("not an Affekt session file")
    if data.get("version") != VERSION:
        raise StateError(f"unsupported session version: {data.get('version')!r}")
    raw_npcs = data.get("npcs")
    if not isinstance(raw_npcs, dict):
        raise StateError("npcs: expected an object")
    known = set(_npc_ids(engine))
    clean: Dict[str, Any] = {}
    for npc_id, entry in raw_npcs.items():
        if npc_id not in known:
            raise StateError(f"unknown character: {str(npc_id)[:40]!r}")
        if not isinstance(entry, dict):
            raise StateError(f"{npc_id}: expected an object")
        vec = _list(entry.get("vector"), 4, f"{npc_id}.vector")
        if len(vec) != 4:
            raise StateError(f"{npc_id}.vector: expected 4 numbers")
        clean[npc_id] = {
            "vector": [_num(v, lo, hi, f"{npc_id}.vector") for v, (lo, hi) in zip(vec, _RANGES)],
            "last_reply": _text(entry.get("last_reply", ""), f"{npc_id}.last_reply"),
            "short_term": [{"phrase": _text(e.get("phrase") if isinstance(e, dict) else None, f"{npc_id}.short_term"),
                            "age_s": _num(e.get("age_s", 0), 0.0, MAX_AGE, f"{npc_id}.short_term")}
                           for e in _list(entry.get("short_term", []), MAX_SHORT, f"{npc_id}.short_term")],
            "medium_term": [{"description": _text(e.get("description") if isinstance(e, dict) else None, f"{npc_id}.medium_term"),
                             "importance": _num(e.get("importance", 0), 0.0, 1.5, f"{npc_id}.medium_term"),
                             "age_s": _num(e.get("age_s", 0), 0.0, MAX_AGE, f"{npc_id}.medium_term")}
                            for e in _list(entry.get("medium_term", []), MAX_MEDIUM, f"{npc_id}.medium_term")],
            "long_term": [{"description": _text(e.get("description") if isinstance(e, dict) else None, f"{npc_id}.long_term"),
                           "importance": _num(e.get("importance", 0), 0.0, 1.5, f"{npc_id}.long_term"),
                           "age_s": _num(e.get("age_s", 0), 0.0, MAX_AGE, f"{npc_id}.long_term")}
                          for e in _list(entry.get("long_term", []), MAX_LONG, f"{npc_id}.long_term")],
        }
    return clean


# ------------------------------------------------------------------ import (validate first, then replace)
def import_state(engine, data: Any) -> Dict[str, int]:
    clean = validate_state(engine, data)  # raises before anything is touched
    now = time.time()

    # 1) emotions: everyone back to baseline, then the saved vectors
    for npc_id in _npc_ids(engine):
        p = engine.personality_for(npc_id)
        p.vector = list(clean[npc_id]["vector"]) if npc_id in clean else list(p.baseline)
        node = engine.graph.get_node(npc_id)
        if node is not None:
            node.properties["current_mood"] = p.mood()
    engine._last_reply = {n: e["last_reply"] for n, e in clean.items() if e["last_reply"]}

    # 2) memory and the graph nodes that mirror it
    for node in list(engine.graph.get_nodes_by_label("MediumTermEvent")) + list(engine.graph.get_nodes_by_label("LongTermMemory")):
        engine.graph.remove_node(node.node_id)
    mem = engine.memory
    mem.short_term_buffer = {}
    mem.medium_term_memory = []
    mem.long_term_memory = []
    events = memories = concepts = 0
    has_player = engine.graph.get_node("player_01") is not None
    for npc_id, e in clean.items():
        buf = mem.short_term_buffer.setdefault(npc_id, deque(maxlen=mem.short_term_capacity))
        for s in e["short_term"]:
            buf.append({"npc_id": npc_id, "phrase": s["phrase"], "timestamp": now - s["age_s"], "type": "short_term"})
        for m in e["medium_term"]:
            events += 1
            node_id = f"event_{npc_id}_{events}"
            engine.graph.add_node(node_id, "MediumTermEvent", description=m["description"],
                                  timestamp_created=now - m["age_s"], importance=m["importance"])
            engine.graph.add_edge(npc_id, "EXPERIENCED", node_id)
            if has_player:
                engine.graph.add_edge("player_01", "CAUSED", node_id)
            mem.medium_term_memory.append({"npc_id": npc_id, "fact_type": "memory",
                                           "data": {"description": m["description"], "node_id": node_id},
                                           "timestamp": now - m["age_s"], "importance": m["importance"]})
        for l in e["long_term"]:
            memories += 1
            concepts += 1
            node_id = f"memory_{npc_id}_{memories}"
            engine.graph.add_node(node_id, "LongTermMemory", concept=l["description"], strength=l["importance"])
            engine.graph.add_edge(npc_id, "REMEMBERED", node_id)
            mem.long_term_memory.append({"npc_id": npc_id, "concept_id": f"CONCEPT_{npc_id}_{concepts}",
                                         "data": {"key_info": {"description": l["description"]},
                                                  "original_importance": l["importance"],
                                                  "source_timestamp": now - l["age_s"]},
                                         "consolidated_at": now - l["age_s"], "type": "long_term"})
    engine._event_counter, engine._memory_counter, mem._concept_counter = events, memories, concepts
    return {"characters": len(clean), "short_term": sum(len(e["short_term"]) for e in clean.values()),
            "medium_term": events, "long_term": memories}
