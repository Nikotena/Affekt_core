"""
Neo4j Management (Graph Database)
Passive infrastructure: stores the unified network of relationships, facts, and concepts.

Architecture:
- Primary store is an in-memory graph for zero-latency access during gameplay.
- Neo4j is an OPTIONAL persistence layer for save/load and offline analysis.
- If the neo4j driver is not installed, the module degrades gracefully to
  in-memory-only mode (all methods still work, nothing is persisted to disk).
"""

import json
import math
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

try:
    from neo4j import GraphDatabase as _Neo4jDriver
    _NEO4J_AVAILABLE = True
except ImportError:
    _Neo4jDriver = None
    _NEO4J_AVAILABLE = False


# Schema definitions (from pages/Neo4j Management.md)
NODE_LABELS = {
    "NPC": ["name", "role", "current_mood"],
    "Player": ["name", "location", "inventory"],
    "Location": ["name", "region", "unlock_requirement"],
    "Item": ["name", "price", "rarity"],
    "MediumTermEvent": ["description", "timestamp_created", "importance"],
    "LongTermMemory": ["concept", "emotional_valence", "strength"],
    "Concept": ["name", "category"],
}

RELATIONSHIP_TYPES = {
    "LOCATED_AT": "Connects any entity to a Location",
    "OWNS": "Links Player or NPC to an Item",
    "KNOWS": "Social connection between characters",
    "EXPERIENCED": "Links an entity to a MediumTermEvent",
    "CAUSED": "Indicates which entity triggered an event",
    "REMEMBERED": "Connects to a LongTermMemory consolidation",
    "MENTIONS": "Links an event or memory to a Concept",
}


class _GraphNode:
    """An entity in the in-memory graph (matches a Neo4j node)."""

    __slots__ = ("node_id", "label", "properties")

    def __init__(self, node_id: str, label: str, properties: Dict[str, Any]):
        self.node_id = node_id
        self.label = label
        self.properties = properties

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "label": self.label,
            "properties": dict(self.properties),
        }


class GraphDatabase:
    """
    Unified graph store for all NPCs, the player, objects, locations, and concepts.

    Acts as the passive persistence layer described in the Neo4j Management module.
    The in-memory graph is always authoritative at runtime; Neo4j is synced on
    save points and graceful shutdown.
    """

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 auto_connect: bool = False):
        # In-memory primary graph
        self.nodes: Dict[str, _GraphNode] = {}
        # adjacency: node_id -> list of (rel_type, target_id, props)
        self.edges: Dict[str, List[tuple]] = defaultdict(list)
        # reverse adjacency: node_id -> list of (rel_type, source_id), for undirected traversal
        self._incoming: Dict[str, List[tuple]] = defaultdict(list)
        # nodes removed since the last sync, so Neo4j forgets them too
        self._removed: Set[str] = set()

        # Optional Neo4j connection
        self._driver = None
        self.uri = uri
        self.user = user
        self.password = password

        if auto_connect:
            self.connect()

    # ------------------------------------------------------------------
    # Connection management (optional persistence layer)
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """Attempt to connect to Neo4j. Returns True on success."""
        if not _NEO4J_AVAILABLE:
            print("[GraphDatabase] neo4j driver not installed; running in-memory only")
            return False
        try:
            self._driver = _Neo4jDriver.driver(
                self.uri, auth=(self.user, self.password)
            )
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"[GraphDatabase] Neo4j connection failed: {e}; running in-memory only")
            self._driver = None
            return False

    def close(self) -> None:
        """Sync to Neo4j (if connected) and release the driver."""
        if self._driver is not None:
            self.sync_to_neo4j()
            self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Node & edge operations (in-memory, zero latency)
    # ------------------------------------------------------------------
    def add_node(self, node_id: str, label: str, **properties) -> _GraphNode:
        if label not in NODE_LABELS:
            raise ValueError(f"Unknown node label: {label}")
        node = _GraphNode(node_id, label, dict(properties))
        self.nodes[node_id] = node
        self._removed.discard(node_id)
        return node

    def get_node(self, node_id: str) -> Optional[_GraphNode]:
        return self.nodes.get(node_id)

    def get_nodes_by_label(self, label: str) -> List[_GraphNode]:
        return [n for n in self.nodes.values() if n.label == label]

    def add_edge(self, source_id: str, rel_type: str, target_id: str, **properties) -> None:
        if rel_type not in RELATIONSHIP_TYPES:
            raise ValueError(f"Unknown relationship type: {rel_type}")
        if source_id not in self.nodes or target_id not in self.nodes:
            raise KeyError("Both source and target nodes must exist before adding an edge")
        # Idempotent: re-adding the same (source, type, target) updates its properties.
        for rel, tgt, props in self.edges[source_id]:
            if rel == rel_type and tgt == target_id:
                props.update(properties)
                return
        self.edges[source_id].append((rel_type, target_id, dict(properties)))
        self._incoming[target_id].append((rel_type, source_id))

    def remove_node(self, node_id: str) -> None:
        """Delete a node and every edge touching it (used when memories are forgotten)."""
        if self.nodes.pop(node_id, None) is not None:
            self._removed.add(node_id)
        self.edges.pop(node_id, None)
        self._incoming.pop(node_id, None)
        for src in list(self.edges):
            self.edges[src] = [e for e in self.edges[src] if e[1] != node_id]
        for tgt in list(self._incoming):
            self._incoming[tgt] = [e for e in self._incoming[tgt] if e[1] != node_id]

    def get_neighbors(self, node_id: str,
                      rel_type: Optional[str] = None) -> List[_GraphNode]:
        result = []
        for rel, target_id, _props in self.edges.get(node_id, []):
            if rel_type is None or rel == rel_type:
                target = self.nodes.get(target_id)
                if target is not None:
                    result.append(target)
        return result

    def subgraph_for(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Extract a flat subgraph within `depth` hops of a node, following edges in
        both directions. Only edges whose endpoints are both included are returned.
        Used to assemble LLM context.
        """
        if node_id not in self.nodes:
            return {"nodes": [], "edges": []}

        visited = {node_id}
        frontier = [node_id]
        for _ in range(depth):
            next_frontier = []
            for current in frontier:
                linked = [t for _r, t, _p in self.edges.get(current, [])]
                linked += [s for _r, s in self._incoming.get(current, [])]
                for other in linked:
                    if other in self.nodes and other not in visited:
                        visited.add(other)
                        next_frontier.append(other)
            frontier = next_frontier

        edges = [
            {"source": src, "type": rel, "target": tgt, "properties": props}
            for src in visited
            for rel, tgt, props in self.edges.get(src, [])
            if tgt in visited
        ]
        return {"nodes": [self.nodes[n].to_dict() for n in visited], "edges": edges}

    # ------------------------------------------------------------------
    # Cypher-style persistence (save/load)
    # ------------------------------------------------------------------
    def sync_to_neo4j(self) -> bool:
        """Push the in-memory graph to Neo4j. No-op if not connected."""
        if self._driver is None:
            return False

        with self._driver.session() as session:
            for node_id in sorted(self._removed):  # forgotten memories must not come back on load
                session.run("MATCH (n {id: $id}) DETACH DELETE n", id=node_id)
            self._removed.clear()
            for node in self.nodes.values():
                session.run(
                    f"MERGE (n:`{node.label}` {{id: $id}}) SET n += $props",
                    id=node.node_id, props=node.properties,
                )
            for source_id, rels in self.edges.items():
                for rel_type, target_id, props in rels:
                    session.run(
                        "MATCH (a {id: $src}), (b {id: $dst}) "
                        f"MERGE (a)-[r:`{rel_type}`]->(b) SET r += $props",
                        src=source_id, dst=target_id, props=props,
                    )
        return True

    def load_from_neo4j(self) -> bool:
        """Load the graph from Neo4j into memory. No-op if not connected."""
        if self._driver is None:
            return False

        with self._driver.session() as session:
            for row in session.run(
                "MATCH (n) RETURN n.id AS id, labels(n) AS labels, properties(n) AS props"
            ):
                label = row["labels"][0] if row["labels"] else "Concept"
                props = {k: v for k, v in row["props"].items() if k != "id"}
                if label not in NODE_LABELS:  # not one of ours: skip it rather than abort the whole load
                    print(f"[GraphDatabase] skipping node {row['id']!r} with unknown label {label!r}")
                    continue
                self.add_node(row["id"], label, **props)
            for row in session.run(
                "MATCH (a)-[r]->(b) RETURN a.id AS src, type(r) AS rel, "
                "b.id AS dst, properties(r) AS props"
            ):
                try:
                    self.add_edge(row["src"], row["rel"], row["dst"], **row["props"])
                except (ValueError, KeyError):  # unknown relationship, or an endpoint that was skipped
                    print(f"[GraphDatabase] skipping edge {row['src']!r} -{row['rel']}-> {row['dst']!r}")
        return True

    # ------------------------------------------------------------------
    # Batch export for offline analysis
    # ------------------------------------------------------------------
    @staticmethod
    def _cypher_literal(value: Any) -> str:
        """Cypher literal for a property value (JSON string escaping is valid Cypher)."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float) and not math.isfinite(value):
            return "null"  # Cypher has no NaN/Infinity literal
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(GraphDatabase._cypher_literal(v) for v in value) + "]"
        return json.dumps(str(value), ensure_ascii=False)

    @classmethod
    def _cypher_map(cls, props: Dict[str, Any]) -> str:
        return "{" + ", ".join(f"`{k}`: {cls._cypher_literal(v)}" for k, v in props.items()) + "}"

    def export_cypher_batch(self) -> List[str]:
        """Return a list of Cypher statements representing the whole graph."""
        lit = self._cypher_literal
        statements = []
        for node in self.nodes.values():
            statements.append(
                f"MERGE (n:`{node.label}` {{id: {lit(node.node_id)}}}) "
                f"SET n += {self._cypher_map(node.properties)}"
            )
        for source_id, rels in self.edges.items():
            for rel_type, target_id, props in rels:
                statements.append(
                    f"MATCH (a {{id: {lit(source_id)}}}), (b {{id: {lit(target_id)}}}) "
                    f"MERGE (a)-[r:`{rel_type}`]->(b) SET r += {self._cypher_map(props)}"
                )
        return statements

    def is_persistent(self) -> bool:
        """Whether a live Neo4j connection is available."""
        return self._driver is not None