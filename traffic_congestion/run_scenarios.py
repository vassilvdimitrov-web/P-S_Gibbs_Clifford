"""
Headless scenario runner – no GUI required.

Each scenario places N nodes in a circle, connects them as a ring road,
then marks a subset as Entry and a subset as Exit.  The simulation
parameters (safe_distance, spawn probability, bad-driver fraction, etc.)
are swept across a grid and the resulting TrafficMetrics are saved to
  results/<scenario_label>.npz
and a summary CSV is written to  results/summary.csv
"""

import itertools
import os
import csv
import math
import random
import pickle
import multiprocessing as mp
import numpy as np

# ── bring in the simulation code without the GUI ──────────────────────────────
# Stub out pyray before importing graphics_combined so no window is opened.
# This runs in every worker process (spawn model on macOS).

import sys, types

def _install_pyray_stub():
    if "pyray" in sys.modules:
        return
    pyray_stub = types.ModuleType("pyray")
    for _name in [
        "init_window", "close_window", "set_target_fps", "window_should_close",
        "begin_drawing", "end_drawing", "clear_background",
        "draw_circle", "draw_circle_lines", "draw_line", "draw_triangle",
        "draw_rectangle", "draw_rectangle_rec", "draw_rectangle_lines_ex",
        "draw_text", "measure_text",
        "is_mouse_button_pressed", "is_mouse_button_released",
        "is_key_pressed", "get_mouse_position",
        "check_collision_point_circle", "check_collision_point_rec",
        "Vector2", "Rectangle", "Color",
        "WHITE", "BLACK", "RED", "GREEN", "BLUE", "DARKBLUE", "SKYBLUE",
        "GRAY", "DARKGRAY", "ORANGE", "PURPLE",
        "MouseButton", "KeyboardKey",
    ]:
        setattr(pyray_stub, _name, None)

    class _Enum:
        pass
    _mb = _Enum(); _mb.MOUSE_BUTTON_LEFT = 0; _mb.MOUSE_BUTTON_RIGHT = 1
    _kk = _Enum(); _kk.KEY_S = 0; _kk.KEY_L = 1
    pyray_stub.MouseButton = _mb
    pyray_stub.KeyboardKey = _kk
    pyray_stub.Vector2     = lambda *_: None
    pyray_stub.Rectangle   = lambda *_: None
    pyray_stub.Color       = lambda *_: None
    pyray_stub.WHITE       = None
    pyray_stub.BLACK       = None
    sys.modules["pyray"] = pyray_stub

_install_pyray_stub()
import graphics_combined as sim 


# ── ring-road factory ─────────────────────────────────────────────────────────

def make_ring(n_nodes, radius=250, cx=450, cy=300):
    nodes = []
    for k in range(n_nodes):
        angle = 2 * math.pi * k / n_nodes - math.pi / 2
        nodes.append(np.array([cx + radius * math.cos(angle),
                                cy + radius * math.sin(angle)]))
    edges = []
    for k in range(n_nodes):
        nxt = (k + 1) % n_nodes
        edges.append(sim.Edge(sim.QuadraticBezier(k, nxt, nodes)))
    return nodes, edges, {}


def assign_entries_exits(nodes, edges, node_types, n_entries, n_exits):
    n = len(nodes)
    entry_step = n / n_entries if n_entries else 0
    exit_step  = n / n_exits   if n_exits   else 0

    entry_ids = {round(i * entry_step) % n for i in range(n_entries)}
    offset = n // 2
    exit_ids = set()
    for i in range(n_exits):
        idx = (offset + round(i * exit_step)) % n
        while idx in entry_ids or idx in exit_ids:
            idx = (idx + 1) % n
        exit_ids.add(idx)

    node_types.clear()
    for i in entry_ids:
        node_types[i] = sim.EntryNode(spawn_probability=0.05)
    for i in exit_ids:
        node_types[i] = sim.ExitNode(despawn_probability=0.05, demand=1.0)

    return list(entry_ids), list(exit_ids)


# ── simulation ────────────────────────────────────────────────────────────────

def _run_simulation(nodes, edges, node_types, n_steps, spawn_prob,
                    bad_driver_frac, safe_dist, v_maximum, seed):
    """
    Runs entirely inside one worker process — global patches on sim are safe
    because each process has its own copy of the module.
    """
    random.seed(seed)
    np.random.seed(seed)

    sim.safe_distance = safe_dist
    sim.v_max         = v_maximum

    entry_ids   = [i for i, nt in node_types.items() if isinstance(nt, sim.EntryNode)]
    entry_nodes = [sim.EntryNodeHelper(i) for i in entry_ids]

    for i in entry_ids:
        node_types[i].spawn_probability = spawn_prob

    # Local replacement avoids touching the module-level function in other tasks
    def _gen(enodes, edgs, ntypes, probability=0.05):
        for node in enodes:
            if random.random() < probability:
                outgoing = [e for e in edgs if e.bezier.node0 == node.node_id]
                if not outgoing:
                    continue
                start_edge = random.choice(outgoing)
                exit_nodes = [i for i, t in ntypes.items()
                              if t.__class__.__name__ == "ExitNode"]
                weights    = [max(0.0001, ntypes[i].demand) for i in exit_nodes]
                if not exit_nodes:
                    continue
                dest   = random.choices(exit_nodes, weights=weights, k=1)[0]
                is_bad = random.random() < bad_driver_frac
                node.waiting_queue.append(sim.Car(
                    start_edge,
                    random.uniform(2, 5),
                    random.uniform(0.5, 1.5),
                    dest,
                    is_bad_driver=is_bad,
                ))

    cars    = []
    metrics = sim.TrafficMetrics(road_length=1.0, v_max=v_maximum)

    for _ in range(n_steps):
        sim.update_traffic_lights(entry_nodes)
        sim.apply_traffic_lights(cars, entry_nodes, nodes)
        _gen(entry_nodes, edges, node_types, probability=spawn_prob)
        sim.process_node_entries(entry_nodes, cars)
        sim.update_velocities(cars, nodes, edges)
        sim.update_positions(cars, nodes, edges)
        cars = sim.handle_edge_transitions(cars, edges, nodes)

        total_road_length = sum(e.bezier.total_length(nodes) for e in edges)
        metrics.road_length = total_road_length if total_road_length > 0 else 1.0
        metrics.compute(cars, entry_nodes)

    return metrics


# ── per-scenario worker (top-level so multiprocessing can pickle it) ──────────

def run_scenario(job):
    scenario_id = job["id"]
    n_nodes     = job["n_nodes"]
    n_entries   = job["n_entries"]
    n_exits     = job["n_exits"]
    params      = job["params"]
    label       = job["label"]
    n_steps     = job["n_steps"]

    nodes, edges, node_types = make_ring(n_nodes)
    assign_entries_exits(nodes, edges, node_types, n_entries, n_exits)

    metrics = _run_simulation(
        nodes, edges, node_types,
        n_steps=n_steps,
        spawn_prob=params["spawn_prob"],
        bad_driver_frac=params["bad_driver_frac"],
        safe_dist=params["safe_dist"],
        v_maximum=params["v_max"],
        seed=scenario_id,
    )

    # ── save npz ──────────────────────────────────────────────────────────────
    npz_path = os.path.join("results", label + ".npz")
    metrics.save(npz_path)

    # ── save pkl (type-safe format so graphics.py can load it) ────────────────
    serializable_types = {}
    for k, v in node_types.items():
        if isinstance(v, sim.EntryNode):
            serializable_types[k] = {"__type__": "EntryNode",
                                     "spawn_probability": v.spawn_probability}
        elif isinstance(v, sim.ExitNode):
            serializable_types[k] = {"__type__": "ExitNode",
                                     "despawn_probability": v.despawn_probability,
                                     "demand": v.demand}
    pkl_path = os.path.join("results", label + ".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"nodes": nodes, "edges": edges,
                     "node_types": serializable_types}, f)

    # ── summary stats ─────────────────────────────────────────────────────────
    h           = metrics.history
    avg_vel     = np.mean([r["velocity"]  for r in h])
    avg_density = np.mean([r["density"]   for r in h])
    avg_flow    = np.mean([r["flow"]      for r in h])
    avg_queue   = np.mean([r["queue"]     for r in h])
    max_queue   = max(r["max_queue"]      for r in h)
    pct_cong    = 100 * sum(1 for r in h if r["congested"]) / len(h)
    first_cong  = metrics.first_congestion_step()

    return {
        "id":              scenario_id,
        "n_nodes":         n_nodes,
        "n_entries":       n_entries,
        "n_exits":         n_exits,
        **params,
        "avg_velocity":    round(avg_vel,     4),
        "avg_density":     round(avg_density, 6),
        "avg_flow":        round(avg_flow,    4),
        "avg_queue":       round(avg_queue,   4),
        "max_queue":       max_queue,
        "pct_congested":   round(pct_cong,    2),
        "first_cong_step": first_cong if first_cong is not None else -1,
        "npz":             npz_path,
        "pkl":             pkl_path,
        "label":           label,
    }


# ── scenario grid ─────────────────────────────────────────────────────────────

RING_SIZES = [4, 6, 8]
# RING_SIZES = [4]

ENTRY_EXIT_CONFIGS = [
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 3),
    (3, 3),
]

PARAM_GRID = {
    "spawn_prob":      [0.01, 0.03, 0.06],
    "bad_driver_frac": [0.0,  0.10, 0.25],
    "safe_dist":       [4,    10,   20],
    "v_max":           [30.0, 50.0, 70.0],
}

#  
N_STEPS = 3000


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)

    param_keys   = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())

    # Build the full job list up front
    jobs = []
    scenario_id = 0
    for n_nodes in RING_SIZES:
        for (n_entries, n_exits) in ENTRY_EXIT_CONFIGS:
            if n_entries + n_exits > n_nodes:
                continue
            for param_combo in itertools.product(*param_values):
                params = dict(zip(param_keys, param_combo))
                label = (
                    f"ring{n_nodes}"
                    f"_e{n_entries}x{n_exits}"
                    f"_sp{params['spawn_prob']}"
                    f"_bd{params['bad_driver_frac']}"
                    f"_sd{params['safe_dist']}"
                    f"_vm{int(params['v_max'])}"
                )
                jobs.append({
                    "id":       scenario_id,
                    "n_nodes":  n_nodes,
                    "n_entries": n_entries,
                    "n_exits":  n_exits,
                    "params":   params,
                    "label":    label,
                    "n_steps":  N_STEPS,
                })
                scenario_id += 1

    total = len(jobs)
    print(f"Total scenarios: {total}  |  workers: {mp.cpu_count()}")

    summary_rows = []
    with mp.Pool() as pool:
        for done, row in enumerate(pool.imap_unordered(run_scenario, jobs), 1):
            summary_rows.append(row)
            print(f"[{done}/{total}] {row['label']}  "
                  f"vel={row['avg_velocity']:.1f}  "
                  f"cong={row['pct_congested']:.0f}%  "
                  f"q={row['avg_queue']:.1f}")

    # Sort by original id so the CSV is deterministic
    summary_rows.sort(key=lambda r: r["id"])
    # Remove the 'label' helper key before writing
    for r in summary_rows:
        r.pop("label", None)

    csv_path = os.path.join("results", "summary.csv")
    fieldnames = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nDone. Summary written to {csv_path}")


if __name__ == "__main__":
    main()
