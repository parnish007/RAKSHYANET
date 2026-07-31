"""
Tests for TopologyAnalyzer -- Prompt 3.3
Run: pytest backend/tests/test_topology_analyzer.py -v
"""
import time
import pytest
from typing import Dict

from backend.p2p.gossip_protocol import GossipProtocol, PeerNode
from backend.p2p.topology_analyzer import (
    NetworkEdge,
    NodeStatus,
    TopologyAnalyzer,
    TopologySnapshot,
)


# ================================================================== #
#  Network builders (shared by fixtures and tests)                    #
# ================================================================== #

def _build_star(n_leaves: int = 4) -> Dict[str, GossipProtocol]:
    """
    1 hub + n_leaves leaf nodes.
    Hub connects to all leaves; leaves connect only to hub.
    """
    hub = GossipProtocol("hub")
    network: Dict[str, GossipProtocol] = {"hub": hub}

    for i in range(n_leaves):
        leaf_id = f"leaf_{i}"
        leaf    = GossipProtocol(leaf_id)
        hub.add_peer(PeerNode(node_id=leaf_id))
        leaf.add_peer(PeerNode(node_id="hub"))
        network[leaf_id] = leaf

    for proto in network.values():
        proto.set_network(network)

    return network


def _build_line(n: int = 5) -> Dict[str, GossipProtocol]:
    """
    Linear chain: node_0 -- node_1 -- ... -- node_{n-1}
    """
    ids   = [f"node_{i}" for i in range(n)]
    nodes = {nid: GossipProtocol(nid) for nid in ids}

    for i in range(n - 1):
        nodes[ids[i]].add_peer(PeerNode(node_id=ids[i + 1]))
        nodes[ids[i + 1]].add_peer(PeerNode(node_id=ids[i]))

    for proto in nodes.values():
        proto.set_network(nodes)

    return nodes


def _build_fully_connected(n: int = 4) -> Dict[str, GossipProtocol]:
    """Complete graph K_n — every node connected to every other."""
    ids   = [f"node_{i}" for i in range(n)]
    nodes = {nid: GossipProtocol(nid) for nid in ids}

    for nid, proto in nodes.items():
        for other_id in ids:
            if other_id != nid:
                proto.add_peer(PeerNode(node_id=other_id))

    for proto in nodes.values():
        proto.set_network(nodes)

    return nodes


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture
def star_net():
    return _build_star(n_leaves=4)      # 5 nodes total


@pytest.fixture
def line_net():
    return _build_line(n=5)             # diameter = 4


@pytest.fixture
def full_net():
    return _build_fully_connected(n=4)  # K4, clustering = 1.0


@pytest.fixture
def star_analyzer(star_net) -> TopologyAnalyzer:
    return TopologyAnalyzer(star_net["hub"])


@pytest.fixture
def line_analyzer(line_net) -> TopologyAnalyzer:
    return TopologyAnalyzer(line_net["node_0"])


@pytest.fixture
def full_analyzer(full_net) -> TopologyAnalyzer:
    return TopologyAnalyzer(full_net["node_0"])


# ================================================================== #
#  Snapshot tests                                                      #
# ================================================================== #

class TestSnapshot:
    def test_capture_snapshot_returns_topology_snapshot(self, star_analyzer):
        snap = star_analyzer.capture_snapshot()
        assert isinstance(snap, TopologySnapshot)

    def test_snapshot_has_correct_node_count(self, star_analyzer):
        snap = star_analyzer.capture_snapshot()
        # 1 hub + 4 leaves
        assert len(snap.nodes) == 5

    def test_snapshot_nodes_have_correct_ids(self, star_analyzer, star_net):
        snap = star_analyzer.capture_snapshot()
        snap_ids = {n.node_id for n in snap.nodes}
        assert snap_ids == set(star_net.keys())

    def test_snapshot_has_correct_edge_count(self, star_analyzer):
        snap = star_analyzer.capture_snapshot()
        # Star: hub-leaf_0, hub-leaf_1, hub-leaf_2, hub-leaf_3 = 4 edges
        assert len(snap.edges) == 4

    def test_snapshot_timestamp_is_recent(self, star_analyzer):
        snap = star_analyzer.capture_snapshot()
        assert snap.timestamp != ""
        assert "Z" in snap.timestamp or "+" in snap.timestamp

    def test_snapshot_stored_in_history(self, star_analyzer):
        assert len(star_analyzer.snapshots) == 0
        star_analyzer.capture_snapshot()
        assert len(star_analyzer.snapshots) == 1

    def test_snapshot_history_capped(self):
        net  = _build_star(n_leaves=2)
        analyzer = TopologyAnalyzer(net["hub"], max_snapshots=3)
        for _ in range(5):
            analyzer.capture_snapshot()
        assert len(analyzer.snapshots) == 3

    def test_snapshot_all_nodes_are_node_status(self, star_analyzer):
        snap = star_analyzer.capture_snapshot()
        for n in snap.nodes:
            assert isinstance(n, NodeStatus)

    def test_snapshot_all_edges_are_network_edge(self, star_analyzer):
        snap = star_analyzer.capture_snapshot()
        for e in snap.edges:
            assert isinstance(e, NetworkEdge)

    def test_line_snapshot_has_correct_edge_count(self, line_analyzer):
        snap = line_analyzer.capture_snapshot()
        # Line n=5 → 4 edges
        assert len(snap.edges) == 4

    def test_fully_connected_snapshot_edge_count(self, full_analyzer):
        snap = full_analyzer.capture_snapshot()
        # K4 → C(4,2) = 6 edges
        assert len(snap.edges) == 6


# ================================================================== #
#  Network metrics tests                                               #
# ================================================================== #

class TestMetrics:
    def test_diameter_line_3_nodes_is_2(self):
        net      = _build_line(n=3)
        analyzer = TopologyAnalyzer(net["node_0"])
        assert analyzer.calculate_network_diameter() == 2

    def test_diameter_line_5_nodes_is_4(self, line_analyzer):
        assert line_analyzer.calculate_network_diameter() == 4

    def test_diameter_star_is_2(self, star_analyzer):
        # Leaf → hub → leaf = 2 hops
        assert star_analyzer.calculate_network_diameter() == 2

    def test_diameter_fully_connected_is_1(self, full_analyzer):
        assert full_analyzer.calculate_network_diameter() == 1

    def test_diameter_single_node_is_0(self):
        solo     = GossipProtocol("solo")
        solo.set_network({"solo": solo})
        analyzer = TopologyAnalyzer(solo)
        assert analyzer.calculate_network_diameter() == 0

    def test_clustering_fully_connected_is_1(self, full_analyzer):
        coeff = full_analyzer.calculate_cluster_coefficient()
        assert coeff == pytest.approx(1.0, abs=1e-6)

    def test_clustering_star_is_0(self, star_analyzer):
        coeff = star_analyzer.calculate_cluster_coefficient()
        assert coeff == pytest.approx(0.0, abs=1e-6)

    def test_clustering_in_range(self, line_analyzer):
        coeff = line_analyzer.calculate_cluster_coefficient()
        assert 0.0 <= coeff <= 1.0

    def test_node_degrees_sum_equals_twice_edge_count(self, star_analyzer):
        degrees    = star_analyzer.get_node_degrees()
        total_deg  = sum(degrees.values())
        snap       = star_analyzer.capture_snapshot()
        edge_count = len(snap.edges)
        assert total_deg == 2 * edge_count

    def test_node_degrees_hub_highest_in_star(self, star_analyzer):
        degrees = star_analyzer.get_node_degrees()
        assert degrees["hub"] == max(degrees.values())

    def test_node_degrees_leaves_have_degree_1(self, star_analyzer):
        degrees = star_analyzer.get_node_degrees()
        for nid, deg in degrees.items():
            if nid != "hub":
                assert deg == 1


# ================================================================== #
#  Critical node tests                                                 #
# ================================================================== #

class TestCriticalNodes:
    def test_hub_in_star_is_critical(self, star_analyzer):
        criticals = star_analyzer.find_critical_nodes()
        assert "hub" in criticals

    def test_leaves_in_star_not_critical(self, star_analyzer):
        criticals = star_analyzer.find_critical_nodes()
        for i in range(4):
            assert f"leaf_{i}" not in criticals

    def test_middle_node_in_line_is_critical(self):
        # Line: n0-n1-n2-n3-n4; n1,n2,n3 are articulation points
        net      = _build_line(n=5)
        analyzer = TopologyAnalyzer(net["node_0"])
        criticals = analyzer.find_critical_nodes()
        assert "node_1" in criticals
        assert "node_2" in criticals
        assert "node_3" in criticals

    def test_no_critical_nodes_in_fully_connected(self, full_analyzer):
        criticals = full_analyzer.find_critical_nodes()
        assert criticals == []

    def test_critical_nodes_returns_list(self, star_analyzer):
        result = star_analyzer.find_critical_nodes()
        assert isinstance(result, list)


# ================================================================== #
#  D3 export tests                                                     #
# ================================================================== #

class TestD3Export:
    def test_d3_has_nodes_and_links_keys(self, star_analyzer):
        d3 = star_analyzer.export_d3_format()
        assert "nodes" in d3
        assert "links" in d3

    def test_d3_nodes_have_required_fields(self, star_analyzer):
        d3 = star_analyzer.export_d3_format()
        for node in d3["nodes"]:
            assert "id"       in node
            assert "group"    in node
            assert "size"     in node
            assert "status"   in node
            assert "messages" in node

    def test_d3_links_have_required_fields(self, star_analyzer):
        d3 = star_analyzer.export_d3_format()
        for link in d3["links"]:
            assert "source" in link
            assert "target" in link
            assert "value"  in link

    def test_d3_active_nodes_have_group_1(self, star_analyzer):
        d3 = star_analyzer.export_d3_format()
        for node in d3["nodes"]:
            assert node["group"] == 1   # all active in star network

    def test_d3_node_count_matches_network(self, star_analyzer):
        d3 = star_analyzer.export_d3_format()
        assert len(d3["nodes"]) == 5

    def test_d3_link_count_matches_edges(self, star_analyzer):
        d3 = star_analyzer.export_d3_format()
        # Star has 4 edges, all active
        assert len(d3["links"]) == 4

    def test_d3_node_size_at_least_5(self, star_analyzer):
        d3 = star_analyzer.export_d3_format()
        for node in d3["nodes"]:
            assert node["size"] >= 5

    def test_d3_link_value_at_least_1(self, star_analyzer):
        d3 = star_analyzer.export_d3_format()
        for link in d3["links"]:
            assert link["value"] >= 1


# ================================================================== #
#  Integration tests                                                   #
# ================================================================== #

class TestIntegration:
    def test_5_node_network_full_snapshot(self):
        net      = _build_fully_connected(n=5)
        analyzer = TopologyAnalyzer(net["node_0"])
        snap     = analyzer.capture_snapshot()
        assert len(snap.nodes) == 5
        assert len(snap.edges) == 10   # C(5,2)

    def test_get_latest_snapshot_none_initially(self):
        net      = _build_star(n_leaves=2)
        analyzer = TopologyAnalyzer(net["hub"])
        assert analyzer.get_latest_snapshot() is None

    def test_get_latest_snapshot_after_capture(self, star_analyzer):
        snap = star_analyzer.capture_snapshot()
        assert star_analyzer.get_latest_snapshot() is snap

    def test_full_pipeline_broadcast_then_snapshot(self):
        import random
        random.seed(99)
        net = _build_fully_connected(n=4)
        # Broadcast from node_0
        net["node_0"].broadcast_message({"data": "test"}, "INTELLIGENCE_REPORT")
        analyzer = TopologyAnalyzer(net["node_0"])
        snap     = analyzer.capture_snapshot()
        assert snap.network_diameter == 1
        assert snap.cluster_coefficient == pytest.approx(1.0, abs=1e-6)

    def test_diameter_stored_in_snapshot(self, line_analyzer):
        snap = line_analyzer.capture_snapshot()
        assert snap.network_diameter == 4

    def test_cluster_coefficient_stored_in_snapshot(self, full_analyzer):
        snap = full_analyzer.capture_snapshot()
        assert snap.cluster_coefficient == pytest.approx(1.0, abs=1e-6)
