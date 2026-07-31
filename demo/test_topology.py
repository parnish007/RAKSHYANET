#!/usr/bin/env python3
"""
Smoke test: P2P Network Topology Analyzer -- Prompt 3.3
Creates a 5-node network, broadcasts 10 messages, captures a topology
snapshot, prints graph metrics, and saves topology.json for the frontend.

Run from project root:
    python demo/test_topology.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.p2p.gossip_protocol import (
    MSG_HEARTBEAT,
    MSG_INTELLIGENCE_REPORT,
    MSG_OPTIMIZATION_RESULT,
    GossipProtocol,
    PeerNode,
)
from backend.p2p.topology_analyzer import TopologyAnalyzer

random.seed(42)

SEP  = "-" * 65
SEP2 = "=" * 65

NODE_IDS = [
    "kathmandu_hub",
    "dhulikhel_relay",
    "panauti_relay",
    "banepa_relay",
    "khopasi_relay",
]


# ------------------------------------------------------------------ #
#  Network builder (fully connected, same as Prompt 3.2 smoke test)   #
# ------------------------------------------------------------------ #

def build_network(node_ids: list, fanout: int = 3, ttl: int = 5) -> dict:
    nodes = {
        nid: GossipProtocol(nid, fanout=fanout, ttl=ttl)
        for nid in node_ids
    }
    network_dict = dict(nodes)
    for nid, proto in nodes.items():
        for other_id in node_ids:
            if other_id != nid:
                proto.add_peer(PeerNode(node_id=other_id))
        proto.set_network(network_dict)
    return nodes


# ------------------------------------------------------------------ #
#  Print helpers (ASCII only for Windows cp1252)                       #
# ------------------------------------------------------------------ #

def bar(fraction: float, width: int = 20) -> str:
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_section(title: str) -> None:
    print(f"\n  {SEP}")
    print(f"  {title}")
    print(f"  {SEP}")


def print_metrics(analyzer: TopologyAnalyzer) -> None:
    diameter = analyzer.calculate_network_diameter()
    cluster  = analyzer.calculate_cluster_coefficient()
    degrees  = analyzer.get_node_degrees()
    criticals = analyzer.find_critical_nodes()

    print_section("NETWORK METRICS")
    print(f"  Network diameter   : {diameter}")
    print(f"  Cluster coefficient: {cluster:.4f}  {bar(cluster)}")
    print(f"  Critical nodes     : {criticals if criticals else '(none)'}")

    print(f"\n  {'Node':<22} {'Degree':>6}  {'Degree bar'}")
    print(f"  {'-'*21} {'------':>6}  {'-'*20}")
    max_deg = max(degrees.values()) if degrees else 1
    for nid, deg in sorted(degrees.items(), key=lambda x: -x[1]):
        print(f"  {nid:<22} {deg:>6}  {bar(deg / max_deg)}")


def print_snapshot(snap) -> None:
    print_section("SNAPSHOT SUMMARY")
    print(f"  Timestamp          : {snap.timestamp}")
    print(f"  Nodes              : {len(snap.nodes)}")
    print(f"  Edges              : {len(snap.edges)}")
    print(f"  Total messages     : {snap.total_messages}")
    print(f"  Network diameter   : {snap.network_diameter}")
    print(f"  Cluster coefficient: {snap.cluster_coefficient:.4f}")

    print(f"\n  {'Node ID':<22} {'Peers':>5}  {'Msgs sent':>9}")
    print(f"  {'-'*21} {'-----':>5}  {'---------':>9}")
    for n in sorted(snap.nodes, key=lambda x: x.node_id):
        print(f"  {n.node_id:<22} {n.peer_count:>5}  {n.messages_sent:>9}")

    print(f"\n  {'Edge':<42} {'Active':>6}")
    print(f"  {'-'*41} {'------':>6}")
    for e in sorted(snap.edges, key=lambda x: (x.from_node, x.to_node)):
        edge_str = f"{e.from_node} -- {e.to_node}"
        print(f"  {edge_str:<42} {'yes' if e.is_active else 'no':>6}")


def print_d3_summary(d3: dict) -> None:
    print_section("D3 EXPORT SUMMARY")
    print(f"  Nodes in D3 format : {len(d3['nodes'])}")
    print(f"  Links in D3 format : {len(d3['links'])}")

    print(f"\n  Sample nodes:")
    for node in d3["nodes"][:3]:
        print(f"    id={node['id']:<22} group={node['group']}  "
              f"size={node['size']}  status={node['status']}")

    print(f"\n  Sample links:")
    for link in d3["links"][:3]:
        print(f"    {link['source']:<22} -> {link['target']:<22}  value={link['value']}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    print("\n" + SEP2)
    print("  RAKSHYANET -- TOPOLOGY ANALYZER SMOKE TEST  (Prompt 3.3)")
    print(SEP2)
    print(f"  Nodes   : {len(NODE_IDS)}")
    print(f"  Topology: fully-connected K5 (diameter=1, clustering=1.0)")

    # ---- Build network -------------------------------------------- #
    nodes = build_network(NODE_IDS, fanout=3, ttl=5)
    print(f"\n  Network built: {len(nodes)} nodes, each with {len(NODE_IDS)-1} peers.")

    # ---- Broadcast 10 messages ------------------------------------ #
    print_section("BROADCASTING 10 MESSAGES")
    messages = [
        (NODE_IDS[0], {"event": "landslide",  "severity": 9}, MSG_INTELLIGENCE_REPORT),
        (NODE_IDS[1], {"event": "flood",      "severity": 7}, MSG_INTELLIGENCE_REPORT),
        (NODE_IDS[2], {"event": "road_block", "severity": 5}, MSG_INTELLIGENCE_REPORT),
        (NODE_IDS[3], {"result": "optimal",   "welfare": 0.93}, MSG_OPTIMIZATION_RESULT),
        (NODE_IDS[4], {"node": NODE_IDS[4],   "active": 4}, MSG_HEARTBEAT),
        (NODE_IDS[0], {"event": "aftershock", "severity": 6}, MSG_INTELLIGENCE_REPORT),
        (NODE_IDS[1], {"result": "nash",       "welfare": 0.88}, MSG_OPTIMIZATION_RESULT),
        (NODE_IDS[2], {"node": NODE_IDS[2],   "active": 4}, MSG_HEARTBEAT),
        (NODE_IDS[3], {"event": "supply_low", "severity": 4}, MSG_INTELLIGENCE_REPORT),
        (NODE_IDS[4], {"event": "road_clear", "severity": 3}, MSG_INTELLIGENCE_REPORT),
    ]

    for i, (sender_id, payload, mtype) in enumerate(messages, 1):
        msg = nodes[sender_id].broadcast_message(payload, mtype)
        print(f"  [{i:02d}] {sender_id:<22}  type={mtype:<25} id={msg.message_id[:10]}...")

    total_stored = sum(len(p.message_store) for p in nodes.values())
    print(f"\n  Messages broadcast : 10")
    print(f"  Total msg-copies in network : {total_stored}")

    # ---- Create analyzer and capture snapshot --------------------- #
    analyzer = TopologyAnalyzer(nodes[NODE_IDS[0]], max_snapshots=5)

    # Print metrics (without capturing snapshot yet)
    print_metrics(analyzer)

    # Capture formal snapshot
    snap = analyzer.capture_snapshot()
    print_snapshot(snap)

    # ---- Export D3 format ----------------------------------------- #
    d3 = analyzer.export_d3_format()
    print_d3_summary(d3)

    # ---- Save topology.json --------------------------------------- #
    out_path = Path(__file__).parent / "topology.json"
    out_path.write_text(json.dumps(d3, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")

    # ---- Snapshot history ----------------------------------------- #
    print_section("SNAPSHOT HISTORY")
    print(f"  Snapshots captured : {len(analyzer.snapshots)}")
    latest = analyzer.get_latest_snapshot()
    print(f"  Latest timestamp   : {latest.timestamp}")

    # ---- Assertions ----------------------------------------------- #
    print_section("ASSERTIONS")

    assert len(snap.nodes) == 5,          f"Expected 5 nodes, got {len(snap.nodes)}"
    print("  [OK] snapshot.nodes == 5")

    assert len(snap.edges) == 10,         f"Expected 10 edges (K5), got {len(snap.edges)}"
    print("  [OK] snapshot.edges == 10  (K5 = C(5,2))")

    assert snap.network_diameter == 1,    f"Expected diameter=1, got {snap.network_diameter}"
    print("  [OK] network_diameter == 1")

    assert abs(snap.cluster_coefficient - 1.0) < 1e-5, \
        f"Expected clustering=1.0, got {snap.cluster_coefficient}"
    print("  [OK] cluster_coefficient == 1.0")

    assert len(d3["nodes"]) == 5,         f"D3 nodes count wrong: {len(d3['nodes'])}"
    print("  [OK] d3['nodes'] == 5")

    assert len(d3["links"]) == 10,        f"D3 links count wrong: {len(d3['links'])}"
    print("  [OK] d3['links'] == 10")

    assert all(n["group"] == 1 for n in d3["nodes"]), "Some nodes inactive in D3"
    print("  [OK] all d3 nodes group=1  (active)")

    assert all(lnk["value"] >= 1 for lnk in d3["links"]), "Link value < 1"
    print("  [OK] all d3 links value >= 1")

    criticals = analyzer.find_critical_nodes()
    assert criticals == [], f"K5 should have no critical nodes, got {criticals}"
    print("  [OK] no critical nodes in K5")

    degrees = analyzer.get_node_degrees()
    assert all(d == 4 for d in degrees.values()), \
        f"K5 nodes should all have degree 4, got {degrees}"
    print("  [OK] all nodes have degree 4 in K5")

    assert out_path.exists(), "topology.json was not written"
    print(f"  [OK] topology.json saved ({out_path.stat().st_size} bytes)")

    # ---- Summary -------------------------------------------------- #
    print("\n" + SEP2)
    print("  SUMMARY")
    print(SEP2)
    print(f"  Nodes          : {len(nodes)}")
    print(f"  Edges          : {len(snap.edges)}")
    print(f"  Diameter       : {snap.network_diameter}")
    print(f"  Clustering     : {snap.cluster_coefficient:.4f}")
    print(f"  Critical nodes : {len(criticals)}")
    print(f"  D3 nodes       : {len(d3['nodes'])}")
    print(f"  D3 links       : {len(d3['links'])}")
    print(f"  topology.json  : {out_path.stat().st_size} bytes")

    print("\n" + SEP2)
    print("  SMOKE TEST PASSED")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()
