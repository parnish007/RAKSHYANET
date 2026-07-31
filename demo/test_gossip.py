#!/usr/bin/env python3
"""
Smoke test: P2P Gossip Protocol -- Prompt 3.2
Creates a 5-node network, broadcasts an INTELLIGENCE_REPORT from node_0,
and verifies that all 5 nodes receive it exactly once.

Run from project root:
    python demo/test_gossip.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.p2p.gossip_protocol import (
    MSG_INTELLIGENCE_REPORT,
    MSG_HEARTBEAT,
    GossipProtocol,
    GossipMessage,
    PeerNode,
)

SEP  = "-" * 65
SEP2 = "=" * 65

random.seed(42)   # reproducible peer selection


# ------------------------------------------------------------------ #
#  Network builder                                                     #
# ------------------------------------------------------------------ #

def build_network(node_ids: list, fanout: int = 3, ttl: int = 5) -> dict:
    """
    Create a fully-connected gossip network.
    Each node has all other nodes as active peers and shares
    the same _network dict for in-process delivery.
    """
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
#  Print helpers                                                       #
# ------------------------------------------------------------------ #

def bar(fraction: float, width: int = 20) -> str:
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_network_state(nodes: dict, label: str = "Network State") -> None:
    print(f"\n  {SEP}")
    print(f"  {label}")
    print(f"  {SEP}")
    print(f"  {'Node':<20} {'Peers':>6}  {'Msgs Stored':>11}  {'Has Target?':>11}")
    print(f"  {'-'*19} {'------':>6}  {'-----------':>11}  {'-----------':>11}")
    for nid, proto in nodes.items():
        stats = proto.stats()
        print(f"  {nid:<20} {stats['peers_active']:>6}  {stats['messages_stored']:>11}  "
              f"{'---':>11}")   # filled in after broadcast


def print_propagation_result(
    nodes: dict,
    msg: GossipMessage,
    source_id: str,
) -> None:
    print(f"\n  {SEP}")
    print(f"  PROPAGATION RESULT  (message_id={msg.message_id[:16]}...)")
    print(f"  {SEP}")
    print(f"  {'Node':<20} {'Role':<12} {'Has Msg?':>8}  {'Msgs Total':>10}")
    print(f"  {'-'*19} {'-'*11} {'--------':>8}  {'----------':>10}")
    received_count = 0
    for nid, proto in nodes.items():
        role  = "SOURCE" if nid == source_id else "relay"
        has_it = proto.message_store.contains(msg.message_id)
        if has_it:
            received_count += 1
        flag = "[YES]" if has_it else "[NO] "
        print(f"  {nid:<20} {role:<12} {flag:>8}  "
              f"{len(proto.message_store):>10}")
    print(f"\n  Nodes reached: {received_count}/{len(nodes)}")
    return received_count


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    NODE_IDS  = ["kathmandu_hub", "dhulikhel_relay", "panauti_relay",
                 "banepa_relay",  "khopasi_relay"]
    FANOUT    = 3
    TTL       = 5

    print("\n" + SEP2)
    print("  RAKSHYANET -- GOSSIP PROTOCOL SMOKE TEST  (Prompt 3.2)")
    print(SEP2)
    print(f"  Nodes   : {len(NODE_IDS)}")
    print(f"  Fanout  : {FANOUT}  (forward to {FANOUT} random peers per hop)")
    print(f"  TTL     : {TTL}    (max {FANOUT}^{TTL} = {FANOUT**TTL:,} nodes reachable)")
    print(f"  Topology: fully-connected (each node knows all others)")

    # ---- Build network -------------------------------------------- #
    nodes = build_network(NODE_IDS, fanout=FANOUT, ttl=TTL)
    print(f"\n  Network built: {len(nodes)} nodes, each with "
          f"{len(NODE_IDS)-1} peers registered.")

    # ---- Broadcast INTELLIGENCE_REPORT from source node ----------- #
    source_id = "kathmandu_hub"
    payload = {
        "event_id":           "evt_001",
        "confidence":         0.92,
        "severity":           8,
        "affected_villages":  ["dhulikhel", "panauti"],
        "recommended_action": "AUTO_OPTIMIZE",
        "analysis_summary":   "High-confidence landslide in Dhulikhel.",
    }

    print(f"\n  Broadcasting INTELLIGENCE_REPORT from [{source_id}]...")
    msg = nodes[source_id].broadcast_message(payload, MSG_INTELLIGENCE_REPORT)
    print(f"  Message ID  : {msg.message_id}")
    print(f"  Message type: {msg.message_type}")
    print(f"  Initial TTL : {TTL}  (now at {msg.ttl} after first hop)")
    print(f"  Hops so far : {msg.hops}")

    # ---- Check propagation ---------------------------------------- #
    reached = print_propagation_result(nodes, msg, source_id)

    # ---- Duplicate-send test -------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  DUPLICATE DETECTION TEST")
    print(f"  {SEP}")
    print(f"  Re-sending same message to all nodes...")
    dup_rejected = 0
    for nid, proto in nodes.items():
        result = proto.receive_message(msg)
        if not result:
            dup_rejected += 1
    print(f"  Duplicate rejections: {dup_rejected}/{len(nodes)} nodes rejected (expected all)")

    # ---- Heartbeat broadcast test --------------------------------- #
    print(f"\n  {SEP}")
    print(f"  HEARTBEAT BROADCAST TEST")
    print(f"  {SEP}")
    for i, (nid, proto) in enumerate(nodes.items()):
        hb = proto.broadcast_message(
            {"node_id": nid, "active_peers": proto.stats()["peers_active"],
             "messages_processed": len(proto.message_store)},
            MSG_HEARTBEAT,
        )
        print(f"  [{i+1}] {nid:<20} heartbeat id={hb.message_id[:12]}...")

    # ---- Summary -------------------------------------------------- #
    print(f"\n  {SEP2}")
    print(f"  SUMMARY")
    print(f"  {SEP2}")
    total_msgs = sum(len(p.message_store) for p in nodes.values())
    print(f"  Nodes in network   : {len(nodes)}")
    print(f"  Nodes reached      : {reached}/{len(nodes)}")
    print(f"  Duplicate rejections: {dup_rejected} (should be {len(nodes)})")
    print(f"  Total msgs stored  : {total_msgs} across all nodes")

    # ---- Assertions ----------------------------------------------- #
    assert reached == len(nodes), \
        f"Only {reached}/{len(nodes)} nodes received the message"
    assert dup_rejected == len(nodes), \
        f"Only {dup_rejected}/{len(nodes)} duplicates rejected"

    # Every node must have the source message
    for nid, proto in nodes.items():
        assert proto.message_store.contains(msg.message_id), \
            f"{nid} is missing message {msg.message_id}"

    # No node should have processed the same message twice
    # (verified by the dup_rejected == len(nodes) assertion above)

    print(f"\n  {SEP2}")
    print(f"  SMOKE TEST PASSED")
    print(f"  {SEP2}\n")


if __name__ == "__main__":
    main()
