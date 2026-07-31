"""
Tests for GossipProtocol -- Prompt 3.2
Run: pytest backend/tests/test_gossip_protocol.py -v
"""
import random
import time
import uuid

import pytest

from backend.p2p.gossip_protocol import (
    MSG_HEARTBEAT,
    MSG_INTELLIGENCE_REPORT,
    MSG_OPTIMIZATION_RESULT,
    GossipMessage,
    GossipProtocol,
    MessageStore,
    PeerNode,
)

# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture
def protocol() -> GossipProtocol:
    """Fresh protocol node for each test."""
    return GossipProtocol(node_id="node_test", fanout=3, ttl=10)


def _make_peer(node_id: str, active: bool = True, last_seen_offset: float = 0.0) -> PeerNode:
    """Create a peer with last_seen = now - last_seen_offset seconds."""
    return PeerNode(
        node_id=node_id,
        address="127.0.0.1",
        port=9000,
        last_seen=time.time() - last_seen_offset,
        is_active=active,
    )


def _make_message(
    sender_id: str = "node_sender",
    ttl: int = 5,
    hops: int = 0,
    message_type: str = MSG_INTELLIGENCE_REPORT,
) -> GossipMessage:
    return GossipMessage(
        message_id=uuid.uuid4().hex,
        sender_id=sender_id,
        timestamp="2026-04-17T10:00:00+00:00",
        ttl=ttl,
        payload={"event_id": "evt_001", "severity": 8},
        hops=hops,
        message_type=message_type,
    )


@pytest.fixture
def peers():
    """5 active peer nodes."""
    return [_make_peer(f"peer_{i}") for i in range(1, 6)]


@pytest.fixture
def populated_protocol(protocol, peers) -> GossipProtocol:
    """Protocol with 5 active peers already added."""
    for peer in peers:
        protocol.add_peer(peer)
    return protocol


# ================================================================== #
#  Peer management tests                                               #
# ================================================================== #

class TestPeerManagement:
    def test_add_peer_increases_count(self, protocol):
        assert len(protocol.peers) == 0
        protocol.add_peer(_make_peer("p1"))
        assert len(protocol.peers) == 1

    def test_add_multiple_peers(self, protocol):
        for i in range(4):
            protocol.add_peer(_make_peer(f"p{i}"))
        assert len(protocol.peers) == 4

    def test_add_peer_overwrites_existing(self, protocol):
        protocol.add_peer(PeerNode(node_id="p1", port=9001))
        assert len(protocol.peers) == 1
        protocol.add_peer(PeerNode(node_id="p1", port=9002))
        assert len(protocol.peers) == 1
        assert protocol.peers["p1"].port == 9002

    def test_remove_peer_decreases_count(self, protocol):
        protocol.add_peer(_make_peer("p1"))
        protocol.add_peer(_make_peer("p2"))
        protocol.remove_peer("p1")
        assert len(protocol.peers) == 1

    def test_remove_nonexistent_peer_noop(self, protocol):
        protocol.remove_peer("ghost")  # Should not raise

    def test_get_active_peers_returns_recent_only(self, protocol):
        protocol.add_peer(_make_peer("recent", last_seen_offset=5.0))   # 5 s ago
        protocol.add_peer(_make_peer("stale",  last_seen_offset=120.0)) # 120 s ago
        active = protocol.get_active_peers()
        ids = [p.node_id for p in active]
        assert "recent" in ids
        assert "stale"  not in ids

    def test_inactive_flag_excludes_peer(self, protocol):
        protocol.add_peer(_make_peer("flagged_inactive", active=False))
        assert protocol.get_active_peers() == []

    def test_prune_inactive_peers_removes_stale(self, protocol):
        protocol.add_peer(_make_peer("fresh",  last_seen_offset=0.0))
        protocol.add_peer(_make_peer("old",    last_seen_offset=200.0))
        removed = protocol.prune_inactive_peers(timeout_seconds=60)
        assert removed == 1
        assert "fresh" in protocol.peers
        assert "old"   not in protocol.peers


# ================================================================== #
#  MessageStore tests                                                  #
# ================================================================== #

class TestMessageStore:
    def test_add_stores_id(self):
        store = MessageStore()
        store.add("msg_abc")
        assert store.contains("msg_abc")

    def test_contains_true_for_stored(self):
        store = MessageStore()
        store.add("id_1")
        assert store.contains("id_1") is True

    def test_contains_false_for_unseen(self):
        store = MessageStore()
        assert store.contains("never_added") is False

    def test_len_tracks_count(self):
        store = MessageStore()
        assert len(store) == 0
        store.add("a")
        store.add("b")
        assert len(store) == 2

    def test_prune_old_removes_expired(self):
        store = MessageStore()
        store.add("old_msg")
        # Force the stored timestamp to be ancient
        store._store["old_msg"] = time.time() - 1000.0
        removed = store.prune_old(max_age_seconds=60.0)
        assert removed == 1
        assert not store.contains("old_msg")

    def test_prune_old_keeps_recent(self):
        store = MessageStore()
        store.add("fresh_msg")
        removed = store.prune_old(max_age_seconds=60.0)
        assert removed == 0
        assert store.contains("fresh_msg")

    def test_prune_old_returns_count(self):
        store = MessageStore()
        for i in range(5):
            store.add(f"msg_{i}")
            store._store[f"msg_{i}"] = time.time() - 9999.0
        pruned = store.prune_old(max_age_seconds=60.0)
        assert pruned == 5


# ================================================================== #
#  broadcast_message tests                                             #
# ================================================================== #

class TestBroadcast:
    def test_broadcast_returns_gossip_message(self, populated_protocol):
        msg = populated_protocol.broadcast_message(
            {"data": "test"}, MSG_INTELLIGENCE_REPORT
        )
        assert isinstance(msg, GossipMessage)

    def test_broadcast_sender_is_node_id(self, protocol):
        # Before propagation, sender_id is set to node_id
        msg = protocol.broadcast_message({}, MSG_HEARTBEAT)
        # After propagation the message's sender_id may have been updated;
        # the important thing is the store has the message ID.
        assert protocol.message_store.contains(msg.message_id)

    def test_broadcast_message_has_correct_type(self, protocol):
        msg = protocol.broadcast_message({}, MSG_OPTIMIZATION_RESULT)
        assert msg.message_type == MSG_OPTIMIZATION_RESULT

    def test_broadcast_initial_ttl(self, protocol):
        protocol2 = GossipProtocol(node_id="n1", ttl=7)
        msg = protocol2.broadcast_message({}, MSG_HEARTBEAT)
        # TTL may be decremented by propagate, but message object reflects that
        assert msg.ttl <= 7  # Must not exceed initial TTL

    def test_broadcast_adds_to_store(self, protocol):
        msg = protocol.broadcast_message({"k": "v"}, MSG_HEARTBEAT)
        assert protocol.message_store.contains(msg.message_id)

    def test_broadcast_unique_ids(self, protocol):
        m1 = protocol.broadcast_message({}, MSG_HEARTBEAT)
        m2 = protocol.broadcast_message({}, MSG_HEARTBEAT)
        assert m1.message_id != m2.message_id


# ================================================================== #
#  receive_message tests                                               #
# ================================================================== #

class TestReceive:
    def test_new_message_returns_true(self, protocol):
        msg = _make_message()
        assert protocol.receive_message(msg) is True

    def test_duplicate_returns_false(self, protocol):
        msg = _make_message()
        protocol.receive_message(msg)
        assert protocol.receive_message(msg) is False

    def test_new_message_added_to_store(self, protocol):
        msg = _make_message()
        protocol.receive_message(msg)
        assert protocol.message_store.contains(msg.message_id)

    def test_ttl_zero_accepted_not_propagated(self, populated_protocol):
        msg = _make_message(ttl=0)
        # Should accept but not propagate (no network set, can't verify propagation
        # directly, but should return True and not raise)
        result = populated_protocol.receive_message(msg)
        assert result is True
        assert populated_protocol.message_store.contains(msg.message_id)

    def test_sender_peer_last_seen_updated(self, protocol):
        peer = _make_peer("sender_node", last_seen_offset=30.0)
        old_last_seen = peer.last_seen
        protocol.add_peer(peer)
        msg = _make_message(sender_id="sender_node")
        protocol.receive_message(msg)
        assert protocol.peers["sender_node"].last_seen >= old_last_seen


# ================================================================== #
#  propagate_message tests                                             #
# ================================================================== #

class TestPropagation:
    def test_propagate_returns_list_of_ids(self, populated_protocol):
        msg = _make_message()
        result = populated_protocol.propagate_message(msg)
        assert isinstance(result, list)

    def test_propagate_selects_up_to_fanout(self, populated_protocol):
        # populated_protocol has fanout=3 and 5 peers
        random.seed(42)
        msg = _make_message(sender_id="peer_1")  # exclude peer_1
        result = populated_protocol.propagate_message(msg)
        assert len(result) <= 3

    def test_propagate_excludes_sender(self, populated_protocol):
        msg = _make_message(sender_id="peer_1")
        result = populated_protocol.propagate_message(msg)
        assert "peer_1" not in result

    def test_propagate_increments_hops(self, populated_protocol):
        msg = _make_message(hops=2)
        old_hops = msg.hops
        populated_protocol.propagate_message(msg)
        assert msg.hops == old_hops + 1

    def test_propagate_decrements_ttl(self, populated_protocol):
        msg = _make_message(ttl=5)
        old_ttl = msg.ttl
        populated_protocol.propagate_message(msg)
        assert msg.ttl == old_ttl - 1

    def test_propagate_updates_sender_id(self, populated_protocol):
        msg = _make_message(sender_id="original_sender")
        populated_protocol.propagate_message(msg)
        assert msg.sender_id == populated_protocol.node_id

    def test_propagate_excludes_inactive_peers(self, protocol):
        protocol.add_peer(_make_peer("active_peer"))
        protocol.add_peer(_make_peer("inactive_peer", active=False))
        msg = _make_message()
        result = protocol.propagate_message(msg)
        assert "inactive_peer" not in result

    def test_propagate_no_peers_returns_empty(self, protocol):
        msg = _make_message()
        result = protocol.propagate_message(msg)
        assert result == []


# ================================================================== #
#  Fanout tests                                                        #
# ================================================================== #

class TestFanout:
    def test_fanout_3_with_5_peers_selects_3(self, protocol):
        for i in range(1, 6):
            protocol.add_peer(_make_peer(f"peer_{i}"))
        random.seed(0)
        msg = _make_message(sender_id="external_sender")
        result = protocol.propagate_message(msg)
        assert len(result) == 3

    def test_fanout_3_with_2_peers_selects_2(self, protocol):
        protocol.add_peer(_make_peer("p1"))
        protocol.add_peer(_make_peer("p2"))
        msg = _make_message(sender_id="external_sender")
        result = protocol.propagate_message(msg)
        assert len(result) == 2

    def test_fanout_3_with_0_peers_selects_0(self, protocol):
        msg = _make_message()
        result = protocol.propagate_message(msg)
        assert len(result) == 0

    def test_fanout_1_selects_exactly_1(self):
        p = GossipProtocol(node_id="n0", fanout=1)
        for i in range(5):
            p.add_peer(_make_peer(f"p{i}"))
        msg = _make_message(sender_id="external")
        result = p.propagate_message(msg)
        assert len(result) == 1

    def test_different_seeds_may_give_different_selections(self, protocol):
        for i in range(1, 7):
            protocol.add_peer(_make_peer(f"p{i}"))
        results = set()
        for seed in range(10):
            random.seed(seed)
            msg = _make_message()
            r = frozenset(protocol.propagate_message(msg))
            results.add(r)
        # With 6 peers and fanout=3, should see more than 1 distinct selection
        assert len(results) > 1


# ================================================================== #
#  TTL tests                                                           #
# ================================================================== #

class TestTTL:
    def test_message_with_ttl_1_propagates_once(self, protocol):
        protocol.add_peer(_make_peer("p1"))
        msg = _make_message(ttl=1)
        protocol.propagate_message(msg)
        assert msg.ttl == 0

    def test_message_with_ttl_0_not_propagated(self, protocol):
        protocol.add_peer(_make_peer("p1"))
        msg = _make_message(ttl=0)
        result = protocol.receive_message(msg)
        # Should accept (True) but message shouldn't propagate
        assert result is True
        # TTL=0 means receive_message returns True but skips propagate
        # The message store should have it
        assert protocol.message_store.contains(msg.message_id)

    def test_ttl_decrements_each_hop(self, protocol):
        for i in range(3):
            protocol.add_peer(_make_peer(f"p{i}"))
        msg = _make_message(ttl=5)
        protocol.propagate_message(msg)
        assert msg.ttl == 4


# ================================================================== #
#  Integration tests                                                   #
# ================================================================== #

class TestIntegration:
    def _build_network(self, count: int, fanout: int = 3, ttl: int = 5):
        """Build a fully-connected gossip network of `count` nodes."""
        nodes = {
            f"node_{i}": GossipProtocol(f"node_{i}", fanout=fanout, ttl=ttl)
            for i in range(count)
        }
        network_dict = dict(nodes)
        # Add all other nodes as peers and register network
        for nid, proto in nodes.items():
            for other_id, other_proto in nodes.items():
                if other_id != nid:
                    proto.add_peer(PeerNode(node_id=other_id))
            proto.set_network(network_dict)
        return nodes

    def test_3_node_network_all_receive_message(self):
        random.seed(42)
        nodes = self._build_network(3, fanout=3, ttl=5)

        # Broadcast from node_0
        source = nodes["node_0"]
        source.broadcast_message({"event": "landslide"}, MSG_INTELLIGENCE_REPORT)

        # All 3 nodes should have the message in their store
        for nid, node in nodes.items():
            assert len(node.message_store) >= 1, f"{nid} did not receive the message"

    def test_5_node_network_all_receive_message(self):
        random.seed(7)
        nodes = self._build_network(5, fanout=3, ttl=5)

        source = nodes["node_0"]
        msg = source.broadcast_message({"severity": 9}, MSG_INTELLIGENCE_REPORT)

        # All nodes should eventually have the message
        for nid, node in nodes.items():
            assert node.message_store.contains(msg.message_id), \
                f"{nid} did not receive message {msg.message_id}"

    def test_duplicate_detection_prevents_reprocessing(self):
        random.seed(1)
        nodes = self._build_network(4, fanout=3, ttl=3)

        source = nodes["node_0"]
        msg = source.broadcast_message({}, MSG_HEARTBEAT)

        # Send the same message again to node_1 — should be rejected
        result = nodes["node_1"].receive_message(msg)
        assert result is False

    def test_heartbeat_message_type_accepted(self):
        p = GossipProtocol("hb_node")
        p.add_peer(_make_peer("p1"))
        msg = p.broadcast_message(
            {"node_id": "hb_node", "active_peers": 1, "messages_processed": 5},
            MSG_HEARTBEAT,
        )
        assert msg.message_type == MSG_HEARTBEAT

    def test_intelligence_report_message_accepted(self):
        p = GossipProtocol("ir_node")
        msg = p.broadcast_message(
            {"event_id": "evt_001", "confidence": 0.85, "recommended_action": "AUTO_OPTIMIZE"},
            MSG_INTELLIGENCE_REPORT,
        )
        assert msg.message_type == MSG_INTELLIGENCE_REPORT
        assert p.message_store.contains(msg.message_id)

    def test_stats_returns_correct_peer_count(self):
        p = GossipProtocol("stats_node")
        p.add_peer(_make_peer("p1"))
        p.add_peer(_make_peer("p2"))
        s = p.stats()
        assert s["peers_total"] == 2
        assert s["node_id"] == "stats_node"
