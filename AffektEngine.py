#!/usr/bin/env python3
"""
Affekt Engine - Main Engine Controller
Coordinates all modules and implements the 80/20 architecture
"""

import glob
import os
import time
from typing import Dict, List, Optional

from modules.VectorialPersonality import (CHOLERIC, MELANCHOLIC, NORMAL, PHLEGMATIC, SANGUINE,
                                          VectorialPersonality)
from modules.MemorySet import MemorySet
from modules.ContextFunction import ContextFunction
from modules.GraphDatabase import GraphDatabase
from modules.FactGuard import guard_price
from modules.IntentLLM import refine as refine_intent
from modules.IntentParser import IntentParser, is_commerce
from modules.LLMLayer import (LLMBackend, LlamaCppBackend, StubBackend, FALLBACK_LINES, style_for,
                              set_example_variants)

# Who reacts how: the bartender shrugs things off, the blacksmith holds a grudge.
NPC_TEMPERAMENTS = {"NPC_0": SANGUINE, "NPC_1": NORMAL, "NPC_2": PHLEGMATIC,
                    "NPC_3": MELANCHOLIC, "NPC_4": CHOLERIC}


def part_of_day(hour: float) -> str:
    """morning 05-12, afternoon 12-18, evening 18-22, night 22-05 (local time)."""
    h = float(hour) % 24
    if h < 5 or h >= 22:
        return "night"
    return "morning" if h < 12 else "afternoon" if h < 18 else "evening"


class AffektEngine:
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 connect_neo4j: bool = False,
                 llm: Optional[LLMBackend] = None, intent_llm=None):
        # Core modules (80/20 architecture: 80% deterministic, 20% AI)
        self.llm: LLMBackend = llm or StubBackend()
        # Optional model (anything with .complete(messages)) that resolves lines the rules call
        # "other". Off by default: it costs one extra API call per ambiguous line.
        self.intent_llm = intent_llm
        self.personalities: Dict[str, VectorialPersonality] = {}
        self.parser = IntentParser()
        self._last_reply: Dict[str, str] = {}
        self.warm_up_stats: Optional[Dict[str, float]] = None
        self.last_guard = "not_checked"  # what the fact guard did to the latest reply
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
            "description": "Everyday life in the town of Affekt",  # sin hora: la hora real la pone `Time:` desde el reloj del jugador
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
    
    def set_time_of_day(self, hour: float) -> str:
        """Tell the engine the player's local hour (0-24). NPCs then know if it is morning, afternoon,
        evening or night, and greet accordingly. Returns the part of the day."""
        part = part_of_day(hour)
        self.context.world_state["part_of_day"] = part
        return part

    def initialize_modules(self):
        """Initialize all modules"""
        self.context.switch_context("chapter_start")

        # Initial context setup
        self.context.set_world_state({
            "location": "Affekt Town",
            "chapter": "Act I",
            "time_of_day": "evening"
        })
        
        print("Affekt Engine initialized successfully")
    
    @property
    def personality(self) -> VectorialPersonality:
        """Default NPC's personality (kept for backwards compatibility)."""
        return self.personality_for("NPC_0")

    def personality_for(self, npc_id: str) -> VectorialPersonality:
        """Each NPC has its own emotional state, created on first use."""
        if npc_id not in self.personalities:
            self.personalities[npc_id] = VectorialPersonality(npc_id, NPC_TEMPERAMENTS.get(npc_id, NORMAL))
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
        self.graph.add_node("player_01", "Player", name="Player", location="town_01")
        # NPCs
        self.graph.add_node("NPC_0", "NPC", name="Alan", role="Bartender", current_mood="neutral")
        self.graph.add_node("NPC_1", "NPC", name="Marta", role="Merchant", current_mood="neutral")
        self.graph.add_node("NPC_2", "NPC", name="Guard", role="Guard", current_mood="neutral")
        self.graph.add_node("NPC_3", "NPC", name="Elena", role="Healer", current_mood="neutral")
        self.graph.add_node("NPC_4", "NPC", name="Kael", role="Blacksmith", current_mood="neutral")
        # Locations
        # The world is a town; each NPC works in its own place (the tavern is only Alan's).
        self.graph.add_node("town_01", "Location", name="Town Square", region="Affekt Town")
        self.graph.add_node("gate_01", "Location", name="Town Gate", region="Affekt Town")
        self.graph.add_node("market_01", "Location", name="Market", region="Affekt Town")
        self.graph.add_node("tavern_01", "Location", name="Tavern", region="Affekt Town")
        self.graph.add_node("clinic_01", "Location", name="Clinic", region="Affekt Town")
        self.graph.add_node("forge_01", "Location", name="Forge", region="Affekt Town")
        # Items
        self.graph.add_node("item_potion", "Item", name="Potion", price=25, rarity="common")
        self.graph.add_node("item_sword", "Item", name="Sword", price=100, rarity="uncommon")
        # Concepts
        self.graph.add_node("concept_debt", "Concept", name="Debt", category="social")
        
        # Relationships
        self.graph.add_edge("player_01", "LOCATED_AT", "town_01")
        self.graph.add_edge("NPC_0", "LOCATED_AT", "tavern_01")
        self.graph.add_edge("NPC_1", "LOCATED_AT", "market_01")
        self.graph.add_edge("NPC_2", "LOCATED_AT", "gate_01")
        self.graph.add_edge("NPC_3", "LOCATED_AT", "clinic_01")
        self.graph.add_edge("NPC_4", "LOCATED_AT", "forge_01")
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
        if self.intent_llm is not None:
            parsed = refine_intent(parsed, self.intent_llm)
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
        self._last_reply[npc_id] = response

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
            "price_guard": self.last_guard,
            "payload": payload,
            "payload_size": len(payload),
            "graph_nodes": len(subgraph["nodes"]),
            "graph_edges": len(subgraph["edges"]),
            "memory_event": event_node_id,
            "timestamp": time.time(),
        }

    def warm_up(self) -> Dict[str, float]:
        """Fill the language model's prompt cache ahead of time. Returns {"prefixes", "seconds"}."""
        self.warm_up_stats = self.llm.warm_up()
        return self.warm_up_stats

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
        if self._last_reply.get(npc_id):
            facts.append("Last reply: " + self._last_reply[npc_id])
        recent = [f"Player earlier: {e['phrase']}" for e in self.memory.get_short_term_context(npc_id)]
        return self.context.build_strict_payload(npc_id, recent, facts=facts), subgraph

    def _generate_response(self, payload: List[str], npc_id: str, action: str,
                           use_llm: bool, mood: str) -> str:
        """LLM phrasing for important moments; deterministic line otherwise or on failure.

        The reply is checked against the engine's prices (modules/FactGuard.py) before it is used.
        """
        fallback = FALLBACK_LINES.get(mood, FALLBACK_LINES["neutral"])
        self.last_guard = "not_checked"
        if use_llm:
            try:
                text = self.llm.generate(payload, action)
                if text:
                    checked = guard_price(text, payload, action, fallback)
                    self.last_guard = checked.status
                    if checked.status != "ok":
                        print(f"[Engine] {npc_id}: reply stated {list(checked.stated)} coins, "
                              f"engine says {list(checked.expected)}; {checked.status}")
                    return checked.text
            except Exception as e:
                print(f"[Engine] LLM failed for {npc_id}: {e}")
        return fallback

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


# ----------------------------------------------------------------------------------
# One-call setup for the demo / cloud server
# ----------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ENV = "AFFEKT_MODEL"            # path to the .gguf, overrides the default search
GPU_LAYERS_ENV = "AFFEKT_GPU_LAYERS"  # layers on the GPU; -1 = all (needs a CUDA build of llama-cpp-python)
DEFAULT_MODEL_GLOBS = (
    "model/gemma-4-E2B-it-Q4_0.gguf", "model/*/gemma-4-E2B-it-Q4_0.gguf",   # preferred (see docs)
    "model/gemma-4*.gguf", "model/*/gemma-4*.gguf",
)


def find_default_model() -> Optional[str]:
    """The Gemma 4 file the demo should use: $AFFEKT_MODEL, else the newest-quality match in ./model."""
    env = os.environ.get(MODEL_ENV)
    if env:
        return env
    for pattern in DEFAULT_MODEL_GLOBS:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            if os.path.getsize(path) > 500_000_000:  # skip truncated downloads
                return path
    return None


def create_engine(model_path: Optional[str] = None, n_gpu_layers: Optional[int] = None,
                  n_threads: Optional[int] = None, cache_mb: int = 512, n_ctx: int = 2048,
                  llm: Optional[LLMBackend] = None, warm_up: bool = False,
                  use_amd_api: Optional[bool] = None, llm_intents: bool = False, **engine_kwargs) -> "AffektEngine":
    """Build a ready-to-use engine with the configuration measured best for Gemma 4:

    two few-shot examples drawn from 5 fixed sets per mood (the prompt prefix repeats, so the
    prompt cache skips it), temperature 0.5 / top_p 0.9 / top_k 40, price stated explicitly.

        engine = create_engine()                       # finds the model, CPU
        engine = create_engine(n_gpu_layers=-1)        # cloud GPU
        reply = engine.process_action("NPC_1", "Hey Marta, how much for a potion?")["response"]

    `warm_up=True` reads every prompt prefix once at startup (about a minute on CPU, seconds on a
    GPU) so the first players do not pay for filling the cache; recommended for the server.

    `use_amd_api` picks the language model: True calls the default model (Gemma-4-31B-IT) on the AMD Token Factory API
    (key AMD_API_KEY in the environment or .env, no GGUF file needed), False uses the local GGUF.
    Left as None it uses the API when no `model_path` is given and a key is available, else local.
    The API backend gives up quickly (30 s, 2 retries) so a slow turn falls back to the engine's
    canned line for the mood instead of freezing the game.

    `llm_intents=True` also lets the AMD API model classify lines the rule parser calls "other"
    (+10 points on held-out lines, one extra call per such line). Off by default.

    Pass `llm=` to inject any backend (tests use StubBackend). Extra keyword arguments go to
    AffektEngine (e.g. connect_neo4j=True). Raises instead of silently falling back to canned
    lines, so a wrong path is noticed before the demo.
    """
    if llm is None and use_amd_api is None and model_path is None:
        from modules.ApiBackend import AMD_KEY_NAME, load_key
        use_amd_api = bool(load_key(key_name=AMD_KEY_NAME))
    if llm is None and use_amd_api:
        from modules.ApiBackend import AMDBackend
        llm = AMDBackend(temperature=0.5, timeout=30.0, retries=2)
        llm.shots = 2
        set_example_variants(5)
    if llm is None:
        path = model_path or find_default_model()
        if not path:
            raise FileNotFoundError(
                f"No Gemma 4 model found. Put a .gguf in ./model, pass model_path=, or set ${MODEL_ENV}.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        if n_gpu_layers is None:
            n_gpu_layers = int(os.environ.get(GPU_LAYERS_ENV, "0"))
        llm = LlamaCppBackend(path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers,
                              temperature=0.5, top_p=0.9, top_k=40, prompt_cache_mb=cache_mb)
        llm.shots = 2
        set_example_variants(5)
    intent_llm = None
    if llm_intents:
        if hasattr(llm, "complete"):
            intent_llm = llm
        else:
            from modules.ApiBackend import AMDBackend
            intent_llm = AMDBackend(temperature=0.0, timeout=30.0, retries=2)
    engine = AffektEngine(llm=llm, intent_llm=intent_llm, **engine_kwargs)
    engine.initialize_modules()
    engine.running = True
    if warm_up:
        engine.warm_up()
    return engine


def main():
    """Main entry point"""
    engine = AffektEngine()
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
