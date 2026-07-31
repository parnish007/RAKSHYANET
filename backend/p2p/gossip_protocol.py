"""
Gossip Protocol -- Prompt 3.2

Peer-to-peer gossip broadcast for resilient intelligence dissemination.
No actual networking in this demo — message delivery is simulated in-memory.
In production: replace the `# In production: send via UDP here` stubs with
real socket sends.

Propagation model
-----------------
Each node, on receiving a new message, forwards it to `fanout` randomly
selected active peers.  TTL decrements at each hop; at TTL=0 the message
is accepted locally but not forwarded.  A MessageStore (backed by a Python
dict for O(1) lookup) prevents reprocessing duplicate message IDs.

Coverage estimate
-----------------
  reach ≈ fanout^ttl
  fanout=3, ttl=5  →   243 nodes
  fanout=3, ttl=10 → 59049 nodes
"""
from __future__ import annotations

import random
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
#  Message type constants                                              #
# ------------------------------------------------------------------ #

MSG_INTELLIGENCE_REPORT = "INTELLIGENCE_REPORT"
MSG_OPTIMIZATION_RESULT = "OPTIMIZATION_RESULT"
MSG_HEARTBEAT           = "HEARTBEAT"

VALID_MESSAGE_TYPES = {MSG_INTELLIGENCE_REPORT, MSG_OPTIMIZATION_RESULT, MSG_HEARTBEAT}


# ================================================================== #
#  Data models                                                         #
# ================================================================== #

class PeerNode(BaseModel):
    """A single peer in the gossip network."""
    node_id:       str
    address:       str = "127.0.0.1"
    port:          int = 9000
    last_seen:     float = Field(default_factory=time.time)
    is_active:     bool  = True
    message_count: int   = 0

    model_config = {"frozen": False}

    def mark_seen(self) -> None:
        """Update last_seen to now."""
        self.last_seen = time.time()


class GossipMessage(BaseModel):
    """A single message propagating through the gossip network."""
    message_id:   str
    sender_id:    str
    timestamp:    str
    ttl:          int        # Time-to-live (hop count remaining)
    payload:      Dict
    hops:         int   = 0  # How many hops have occurred
    message_type: str   = MSG_HEARTBEAT

    model_config = {"frozen": False}

    def forwarded_copy(self, new_sender_id: str) -> "GossipMessage":
        """Return a new message with TTL decremented and hop count incremented."""
        return GossipMessage(
            message_id=self.message_id,
            sender_id=new_sender_id,
            timestamp=self.timestamp,
            ttl=self.ttl - 1,
            payload=self.payload,
            hops=self.hops + 1,
            message_type=self.message_type,
        )


# ================================================================== #
#  MessageStore                                                         #
# ================================================================== #

class MessageStore:
    """
    Deduplication store for gossip messages.

    Backed by a dict {message_id: received_at_unix_time} for O(1) lookup
    and age-based pruning.
    """

    def __init__(self) -> None:
        self._store: Dict[str, float] = {}

    def add(self, message_id: str) -> None:
        """Record that this message ID has been seen."""
        self._store[message_id] = time.time()

    def contains(self, message_id: str) -> bool:
        """Return True if message_id has been seen before."""
        return message_id in self._store

    def prune_old(self, max_age_seconds: float = 300.0) -> int:
        """
        Remove entries older than max_age_seconds.
        Returns the number of entries pruned.
        """
        now = time.time()
        to_remove = [
            mid for mid, ts in self._store.items()
            if (now - ts) > max_age_seconds
        ]
        for mid in to_remove:
            del self._store[mid]
        return len(to_remove)

    def __len__(self) -> int:
        return len(self._store)


# ================================================================== #
#  GossipProtocol                                                      #
# ================================================================== #

class GossipProtocol:
    """
    Gossip broadcast node.

    Maintains a peer list and a deduplication store.  Each call to
    broadcast_message() creates a new GossipMessage and returns the
    peer IDs that would receive it.  receive_message() implements the
    standard gossip receive logic (duplicate check → store → propagate).

    For simulation, set self._network to a Dict[node_id, GossipProtocol]
    before calling broadcast_message() or receive_message(); the protocol
    will automatically deliver to peer nodes in-process.

    Args:
        node_id:             Unique identifier for this node.
        fanout:              How many peers to forward each message to.
        ttl:                 Initial time-to-live for broadcast messages.
        heartbeat_interval:  Seconds between heartbeat sends (not used
                             in demo; kept for API parity).
        active_timeout:      Seconds before a peer is considered inactive.
    """

    def __init__(
        self,
        node_id: str,
        fanout: int = 3,
        ttl: int = 10,
        heartbeat_interval: int = 30,
        active_timeout: int = 60,
    ) -> None:
        self.node_id            = node_id
        self.fanout             = fanout
        self.default_ttl        = ttl
        self.heartbeat_interval = heartbeat_interval
        self.active_timeout     = active_timeout

        self.peers:         Dict[str, PeerNode]     = {}
        self.message_store: MessageStore            = MessageStore()

        # Optional: set by simulation harness so protocol can actually
        # deliver messages to peer nodes in-process.
        self._network: Optional[Dict[str, "GossipProtocol"]] = None

    # -------------------------------------------------------------- #
    #  Peer management                                                #
    # -------------------------------------------------------------- #

    def add_peer(self, peer: PeerNode) -> None:
        """Register a peer.  Overwrites if node_id already exists."""
        self.peers[peer.node_id] = peer

    def remove_peer(self, node_id: str) -> None:
        """Remove a peer by node_id (no-op if not found)."""
        self.peers.pop(node_id, None)

    def get_active_peers(self) -> List[PeerNode]:
        """
        Return peers that are flagged active and whose last_seen is
        within active_timeout seconds.
        """
        now = time.time()
        return [
            p for p in self.peers.values()
            if p.is_active and (now - p.last_seen) <= self.active_timeout
        ]

    def prune_inactive_peers(self, timeout_seconds: Optional[int] = None) -> int:
        """
        Remove peers that are inactive or last seen more than
        timeout_seconds ago.  Returns the count removed.
        """
        limit = timeout_seconds if timeout_seconds is not None else self.active_timeout
        now   = time.time()
        to_remove = [
            nid for nid, p in self.peers.items()
            if not p.is_active or (now - p.last_seen) > limit
        ]
        for nid in to_remove:
            del self.peers[nid]
        return len(to_remove)

    # -------------------------------------------------------------- #
    #  Message creation                                               #
    # -------------------------------------------------------------- #

    def broadcast_message(
        self,
        payload: Dict,
        message_type: str,
    ) -> GossipMessage:
        """
        Create a new message originating from this node and propagate
        to fanout peers.

        Returns the GossipMessage that was created.
        """
        message = GossipMessage(
            message_id=uuid.uuid4().hex,
            sender_id=self.node_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            ttl=self.default_ttl,
            payload=payload,
            hops=0,
            message_type=message_type,
        )
        self.message_store.add(message.message_id)
        self.propagate_message(message)
        return message

    # -------------------------------------------------------------- #
    #  Message receive                                                 #
    # -------------------------------------------------------------- #

    def receive_message(self, message: GossipMessage) -> bool:
        """
        Process an incoming gossip message.

        Returns:
            True  — message was new and accepted (may have been propagated).
            False — message was a duplicate and was dropped.
        """
        # Duplicate check
        if self.message_store.contains(message.message_id):
            return False

        # Accept
        self.message_store.add(message.message_id)

        # Update sender peer's last_seen if we know about them
        sender_peer = self.peers.get(message.sender_id)
        if sender_peer:
            sender_peer.mark_seen()
            sender_peer.message_count += 1

        # TTL check — accept locally but do not propagate
        if message.ttl <= 0:
            return True

        # Propagate onward
        self.propagate_message(message)
        return True

    # -------------------------------------------------------------- #
    #  Propagation                                                     #
    # -------------------------------------------------------------- #

    def propagate_message(self, message: GossipMessage) -> List[str]:
        """
        Forward message to a random subset of active peers.

        Mutates the message in-place (hops +=1, ttl -=1, sender_id updated).
        Also delivers in-process if self._network is set (simulation mode).

        Returns the list of peer node_ids selected for forwarding.

        In production: replace delivery block with UDP send.
        """
        active    = self.get_active_peers()
        available = [p for p in active if p.node_id != message.sender_id]

        selected_count = min(self.fanout, len(available))
        if selected_count == 0:
            # Mutate even with no peers so hops/ttl stay consistent
            message.hops     += 1
            message.ttl      -= 1
            message.sender_id = self.node_id
            return []

        selected_peers = random.sample(available, selected_count)

        # Mutate message for this hop
        message.hops     += 1
        message.ttl      -= 1
        message.sender_id = self.node_id

        propagated_ids = [p.node_id for p in selected_peers]

        # ---- In production: send via UDP here ---- #
        if self._network is not None:
            for peer in selected_peers:
                target = self._network.get(peer.node_id)
                if target is not None:
                    # Deliver a fresh copy so each recipient sees their own TTL
                    copy = message.forwarded_copy(new_sender_id=self.node_id)
                    # Restore before-forwarded TTL for recipient's view
                    # (copy already has ttl = message.ttl - 1 from forwarded_copy)
                    # Actually we want to pass message as-is (already mutated)
                    target.receive_message(
                        GossipMessage(
                            message_id=message.message_id,
                            sender_id=self.node_id,
                            timestamp=message.timestamp,
                            ttl=message.ttl,          # already decremented
                            payload=message.payload,
                            hops=message.hops,
                            message_type=message.message_type,
                        )
                    )
        # ----------------------------------------- #

        return propagated_ids

    # -------------------------------------------------------------- #
    #  Network helpers                                                 #
    # -------------------------------------------------------------- #

    def set_network(self, network: Dict[str, "GossipProtocol"]) -> None:
        """
        Attach an in-process network dict for simulation.
        Keys are node_ids; values are GossipProtocol instances.
        """
        self._network = network

    def stats(self) -> Dict:
        """Return a summary of this node's current state."""
        return {
            "node_id":         self.node_id,
            "peers_total":     len(self.peers),
            "peers_active":    len(self.get_active_peers()),
            "messages_stored": len(self.message_store),
        }
