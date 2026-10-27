import pytest

from kasc_v2.routing import EDGES, GATES, NODES, SECTIONS, edge_id, scenario_estimate, shortest_route, topology


def test_every_supported_gate_to_section_is_reachable():
    for gate in GATES:
        for section in SECTIONS:
            path = shortest_route(gate, section)
            assert path['reachable'], (gate, section)
            assert path['path'][0] == gate
            assert path['path'][-1] == section
            assert path['travel_min'] > 0


def test_all_edges_have_defined_nodes_and_positive_weights():
    for a, b, weight in EDGES:
        assert a in NODES and b in NODES
        assert weight > 0
    assert len(NODES) == len(set(NODES))


def test_closure_removes_a_only_entrance():
    route = shortest_route('G1', 'S101', blocked={edge_id('G1', 'CP1')})
    assert route['reachable'] is False
    assert route['path'] == []
    assert route['travel_min'] is None


def test_congestion_can_change_cost_but_not_create_negative_times():
    base = shortest_route('G1', 'S105')
    busy = shortest_route('G1', 'S105', crowd={'CP1': 25, 'N': 18, 'E': 20})
    assert busy['weighted_cost'] >= busy['travel_min']
    assert base['weighted_cost'] >= 0


def test_nonexistent_nodes_are_rejected():
    with pytest.raises(ValueError):
        shortest_route('G900', 'S101')


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        shortest_route('G1', 'S101', mode='unsafe')


def test_extra_lanes_dont_increase_fluid_queue_wait():
    current = scenario_estimate(140, 25, 2, 2.5, extra_lanes=0)
    add = scenario_estimate(140, 25, 2, 2.5, extra_lanes=2)
    assert add['scenario_wait_min'] <= current['scenario_wait_min']
    assert add['scenario_queue'] <= current['scenario_queue']


def test_invalid_scenario_is_rejected():
    with pytest.raises(ValueError):
        scenario_estimate(10, 5, 2, 3, extra_lanes=-1)


def test_topology_no_duplicate_or_missing_ids():
    graph = topology()
    assert len(graph['nodes']) == len(NODES)
    ids = [edge['id'] for edge in graph['edges']]
    assert len(ids) == len(set(ids))
    assert graph['illustrative'] is True
