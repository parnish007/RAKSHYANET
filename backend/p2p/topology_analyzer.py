"""
Topology Analyzer -- Prompt 3.3

Captures snapshots of the P2P gossip network, computes graph-theoretic
metrics (diameter, clustering coefficient, articulation points), and
exports the topology in D3.js force-directed graph format.

Uses NetworkX for all graph algorithms — no need to reinvent BFS/DFS.

Typical usage
-------------
    from backend.p2p.gossip_protocol import GossipProtocol
    from backend.p2p.topology_analyzer import TopologyAnalyzer

    # Build and connect a network (see GossipProtocol.set_network)
    analyzer = TopologyAnalyzer(hub_protocol)
    snapshot = analyzer.capture_snapshot()
    d3_data  = analyzer.export_d3_format()
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import networkx as nx
from pydantic import BaseModel, Field

from backend.p2p.gossip_protocol import GossipProtocol, PeerNode


# ================================================================== #
#  Data models                                                         #
# ================================================================== #

class NodeStatus(BaseModel):
    """Health and activity snapshot of a single network node."""
    node_id:           str
    is_active:         bool  = True
    peer_count:        int   = 0
    messages_sent:     int   = 0
    messages_received: int   = 0
    last_seen:         float = Field(default_factory=time.time)
    uptime_seconds:    float = 0.0


class NetworkEdge(BaseModel):
    """A peer-to-peer connection between two nodes."""
    from_node:         str
    to_node:           str
    message_count:     int  = 0
    last_message_time: str  = ""
    is_active:         bool = True


class TopologySnapshot(BaseModel):
    """Point-in-time snapshot of the entire gossip network."""
    timestamp:            str
    nodes:                List[NodeStatus]  = Field(default_factory=list)
    edges:                List[NetworkEdge] = Field(default_factory=list)
    total_messages:       int   = 0
    network_diameter:     int   = 0
    cluster_coefficient:  float = 0.0


# ================================================================== #
#  TopologyAnalyzer                                                    #
# ================================================================== #

class TopologyAnalyzer:
    """
    Wraps a GossipProtocol and computes network-level topology metrics.

    If protocol._network is set (via GossipProtocol.set_network), the
    analyzer sees the FULL multi-node network.  Otherwise it sees only
    the star graph formed by the protocol's direct peers.

    Args:
        protocol: The GossipProtocol instance that anchors the view.
        max_snapshots: Maximum snapshot history to keep in memory.
    """

    def __init__(
        self,
        protocol: GossipProtocol,
        max_snapshots: int = 10,
    ) -> None:
        self.protocol      = protocol
        self.max_snapshots = max_snapshots
        self.snapshots:    List[TopologySnapshot] = []

    # -------------------------------------------------------------- #
    #  Internal helpers                                               #
    # -------------------------------------------------------------- #

    def _get_network(self) -> Dict[str, GossipProtocol]:
        """
        Return the full network dict if available, else a single-node
        view {self.protocol.node_id: self.protocol}.
        """
        if self.protocol._network:
            return self.protocol._network
        return {self.protocol.node_id: self.protocol}

    def _build_graph(self) -> nx.Graph:
        """
        Construct an undirected NetworkX graph from peer connections.
        Only adds edges where both nodes are known in the network.
        """
        G   = nx.Graph()
        net = self._get_network()

        for nid in net:
            G.add_node(nid)

        for nid, proto in net.items():
            for peer_id in proto.peers:
                if peer_id in net:       # Only connect to known nodes
                    G.add_edge(nid, peer_id)

        return G

    def _build_edges(self) -> List[NetworkEdge]:
        """
        Enumerate unique undirected edges from peer connections.
        Uses a sorted-tuple key to deduplicate (A-B == B-A).
        """
        net  = self._get_network()
        seen = set()
        edges: List[NetworkEdge] = []
        now_str = datetime.now(timezone.utc).isoformat()

        for nid, proto in net.items():
            for peer_id, peer in proto.peers.items():
                key = tuple(sorted([nid, peer_id]))
                if key in seen:
                    continue
                seen.add(key)
                edges.append(NetworkEdge(
                    from_node=key[0],
                    to_node=key[1],
                    message_count=peer.message_count,
                    last_message_time=now_str,
                    is_active=peer.is_active,
                ))

        return edges

    # -------------------------------------------------------------- #
    #  Public API                                                     #
    # -------------------------------------------------------------- #

    def capture_snapshot(self) -> TopologySnapshot:
        """
        Capture the current network state as a TopologySnapshot.

        Iterates over all known protocols (or just the anchor's peers)
        to build NodeStatus and NetworkEdge lists, then computes metrics.
        """
        net = self._get_network()
        now = time.time()
        nodes: List[NodeStatus] = []

        for nid, proto in net.items():
            status = NodeStatus(
                node_id=nid,
                is_active=True,
                peer_count=len(proto.get_active_peers()),
                messages_sent=len(proto.message_store),
                messages_received=0,     # Production: track separately
                last_seen=now,
                uptime_seconds=0.0,      # Production: track node start time
            )
            nodes.append(status)

        edges    = self._build_edges()
        diameter = self.calculate_network_diameter()
        cluster  = self.calculate_cluster_coefficient()

        snapshot = TopologySnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            nodes=nodes,
            edges=edges,
            total_messages=sum(n.messages_sent for n in nodes),
            network_diameter=diameter,
            cluster_coefficient=cluster,
        )

        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

        return snapshot

    def calculate_network_diameter(self) -> int:
        """
        Longest shortest path between any pair of nodes.

        Returns 0 for disconnected or single-node graphs.
        """
        G = self._build_graph()
        if len(G.nodes) < 2 or not nx.is_connected(G):
            return 0
        return nx.diameter(G)

    def calculate_cluster_coefficient(self) -> float:
        """
        Average local clustering coefficient across all nodes.

        Ranges from 0.0 (star / tree) to 1.0 (fully connected).
        Returns 0.0 for graphs with fewer than 3 nodes.
        """
        G = self._build_graph()
        if len(G.nodes) < 3:
            return 0.0
        return round(nx.average_clustering(G), 6)

    def get_node_degrees(self) -> Dict[str, int]:
        """
        Return the degree (active peer count) of every node.

        In an undirected graph: sum(degrees) == 2 × edge_count.
        """
        G = self._build_graph()
        return dict(G.degree())

    def find_critical_nodes(self) -> List[str]:
        """
        Return nodes whose removal disconnects the network (articulation points).

        Uses NetworkX's O(V+E) algorithm internally.
        Returns an empty list for disconnected or trivial graphs.
        """
        G = self._build_graph()
        if len(G.nodes) < 2 or not nx.is_connected(G):
            return []
        return sorted(nx.articulation_points(G))

    def export_d3_format(self) -> Dict:
        """
        Export the current topology in D3.js force-directed graph format.

        Returns::

            {
              "nodes": [{"id": ..., "group": 1, "size": ..., "status": ..., "messages": ...}],
              "links": [{"source": ..., "target": ..., "value": ...}]
            }

        ``group=1`` for active nodes, ``group=0`` for inactive.
        ``size`` is proportional to peer_count.
        ``value`` (link thickness) is proportional to message_count.
        """
        snapshot = self.capture_snapshot()

        d3_nodes = [
            {
                "id":       n.node_id,
                "group":    1 if n.is_active else 0,
                "size":     max(5, n.peer_count * 2),
                "status":   "active" if n.is_active else "inactive",
                "messages": n.messages_sent,
            }
            for n in snapshot.nodes
        ]

        d3_links = [
            {
                "source": e.from_node,
                "target": e.to_node,
                "value":  max(1, e.message_count),
            }
            for e in snapshot.edges
            if e.is_active
        ]

        return {"nodes": d3_nodes, "links": d3_links}

    def get_latest_snapshot(self) -> Optional[TopologySnapshot]:
        """Return the most recent snapshot, or None if none exist."""
        return self.snapshots[-1] if self.snapshots else None
