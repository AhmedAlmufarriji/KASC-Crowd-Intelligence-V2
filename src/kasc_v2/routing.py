"""Original illustrative venue topology, Dijkstra routing, and policy scenarios.

None of these nodes or edge lengths are claimed to be an accurate physical map
of King Abdullah Sports City. All distances and costs are illustrative.
"""
from __future__ import annotations

import heapq
from math import inf

GATES = [f"G{i}" for i in range(1, 7)]
SECTIONS = [f"S{i}" for i in range(101, 113)]

NODES = {
    "G1": (13, 8), "G2": (50, 4), "G3": (87, 8),
    "G4": (87, 92), "G5": (50, 96), "G6": (13, 92),
    "CP1": (24, 19), "CP2": (50, 14), "CP3": (76, 19),
    "CP4": (76, 81), "CP5": (50, 86), "CP6": (24, 81),
    "N": (50, 26), "E": (74, 50), "S": (50, 74), "W": (26, 50),
    "Z1": (38, 37), "Z2": (62, 37), "Z3": (62, 63), "Z4": (38, 63),
    **{f"S{101+i}": pt for i, pt in enumerate([
        (34, 39), (38, 33), (43, 39),
        (57, 39), (62, 33), (66, 39),
        (57, 61), (62, 67), (66, 61),
        (34, 61), (38, 67), (43, 61),
    ])},
}

# Undirected walkways with illustration-only baseline travel times (minutes).
EDGES = [
    ("G1", "CP1", 2.0), ("G2", "CP2", 2.0), ("G3", "CP3", 2.0),
    ("G4", "CP4", 2.0), ("G5", "CP5", 2.0), ("G6", "CP6", 2.0),
    ("CP1", "W", 3.0), ("CP2", "N", 3.0), ("CP3", "E", 3.0),
    ("CP4", "E", 3.0), ("CP5", "S", 3.0), ("CP6", "W", 3.0),
    ("N", "E", 3.4), ("E", "S", 3.4), ("S", "W", 3.4), ("W", "N", 3.4),
    ("N", "Z1", 2.0), ("N", "Z2", 2.0),
    ("E", "Z2", 2.0), ("E", "Z3", 2.0),
    ("S", "Z3", 2.0), ("S", "Z4", 2.0),
    ("W", "Z4", 2.0), ("W", "Z1", 2.0),
    ("Z1", "Z2", 3.0), ("Z2", "Z3", 3.0),
    ("Z3", "Z4", 3.0), ("Z4", "Z1", 3.0),
    *[("Z1", f"S{i}", 1.0) for i in (101, 102, 103)],
    *[("Z2", f"S{i}", 1.0) for i in (104, 105, 106)],
    *[("Z3", f"S{i}", 1.0) for i in (107, 108, 109)],
    *[("Z4", f"S{i}", 1.0) for i in (110, 111, 112)],
]


def edge_id(a: str, b: str) -> str:
    return "--".join(sorted([a, b]))


def topology() -> dict:
    return {
        "nodes": [{"id": name, "x": xy[0], "y": xy[1],
                   "kind": "gate" if name in GATES else "section" if name in SECTIONS else
                           "checkpoint" if name.startswith("CP") else "zone" if name.startswith("Z") else "concourse"}
                  for name, xy in NODES.items()],
        "edges": [{"from": a, "to": b, "id": edge_id(a, b), "base_min": w} for a, b, w in EDGES],
        "gate_ids": GATES, "section_ids": SECTIONS,
        "illustrative": True,
    }


def shortest_route(start: str, goal: str, *, crowd: dict[str, float] | None = None,
                   blocked: set[str] | None = None, mode: str = "balanced") -> dict:
    if start not in NODES or goal not in NODES:
        raise ValueError("Unknown start or destination")
    if mode not in {"balanced", "shortest"}:
        raise ValueError("Route mode must be balanced or shortest")
    crowd = crowd or {}
    blocked = blocked or set()
    adjacency: dict[str, list] = {node: [] for node in NODES}
    for a, b, w in EDGES:
        if edge_id(a, b) in blocked:
            continue
        for src, dst in ((a, b), (b, a)):
            # Congestion-aware edge costs, never below nonnegative base duration.
            # Penalize slow segments next to nodes with higher wait (illustrative policy).
            factor = 1.0 if mode == "shortest" else 1 + min(3.0, max(0.0, crowd.get(dst, 0)) / 12.0)
            adjacency[src].append((dst, w * factor, w))
    dist = {node: inf for node in NODES}
    dist[start] = 0.0
    prev: dict[str, str] = {}
    q = [(0.0, start)]
    while q:
        cost, node = heapq.heappop(q)
        if cost > dist[node]:
            continue
        if node == goal:
            break
        for neighbor, weighted, base in adjacency[node]:
            new_cost = cost + weighted
            if new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                prev[neighbor] = node
                heapq.heappush(q, (new_cost, neighbor))
    if dist[goal] == inf:
        return {"reachable": False, "path": [], "travel_min": None, "weighted_cost": None,
                "reason": "No safe route exists under the selected closures"}
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    base_lookup = {edge_id(a, b): w for a, b, w in EDGES}
    travel = sum(base_lookup[edge_id(path[i], path[i+1])] for i in range(len(path)-1))
    return {"reachable": True, "path": path, "travel_min": round(travel, 2),
            "weighted_cost": round(dist[goal], 2), "mode": mode, "illustrative": True}


def scenario_estimate(current_queue: float, arrival_rate: float, lanes: int,
                      service_per_lane: float, extra_lanes: int = 0,
                      horizon_min: int = 15) -> dict:
    """Deterministic fluid queue scenario; NOT a validated operational impact forecast."""
    if lanes < 1 or extra_lanes < 0 or service_per_lane <= 0 or horizon_min <= 0:
        raise ValueError("Invalid scenario parameters")
    arrivals = max(0.0, arrival_rate) * horizon_min
    service = (lanes + extra_lanes) * service_per_lane * horizon_min
    queue_after = max(0.0, current_queue + arrivals - service)
    wait_after = queue_after / ((lanes + extra_lanes) * service_per_lane)
    return {"scenario_queue": round(queue_after, 2), "scenario_wait_min": round(wait_after, 2),
            "extra_lanes": extra_lanes, "horizon_min": horizon_min,
            "model": "deterministic illustrative fluid queue"}
