import heapq
import numpy as np
from pyray import *
from dataclasses import dataclass
from enum import Enum, auto

NODE_RADIUS = 14

# ── Node types ────────────────────────────────────────────────────────────────

@dataclass
class EntryNode:
    spawn_probability: float = 0.05

@dataclass
class ExitNode:
    despawn_probability: float = 0.05
    demand: float = 1.0   # relative weight for destination selection

# ── Interaction state ─────────────────────────────────────────────────────────

class DragState(Enum):
    IDLE     = auto()
    DRAGGING = auto()

# ── Adjacency ────────────────────────────────────────────────────────────────

def rebuild_adjacency(nodes, edges):
    """
    Return (incoming, outgoing) where each is a dict {node_idx: [edge, ...]}.
    Call this after any node or edge is added or removed.
    """
    incoming = {i: [] for i in range(len(nodes))}
    outgoing = {i: [] for i in range(len(nodes))}
    for edge in edges:
        outgoing[edge.bezier.node0].append(edge)
        incoming[edge.bezier.node1].append(edge)
    return incoming, outgoing

# ── Routing ───────────────────────────────────────────────────────────────────

def _dijkstra(src, nodes, outgoing):
    """
    Dijkstra from src. Returns {dst: {'path': [edge,...], 'length': float, 'layer': int}}
    where layer is the number of hops (BFS depth) of the shortest-distance path.
    """
    # heap entries: (total_arc_length, tie_breaker, node, path_so_far)
    counter  = 0
    heap     = [(0.0, counter, src, [])]
    visited  = set()
    result   = {}

    while heap:
        d, _, u, path = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u != src:
            result[u] = {'path': path, 'length': d, 'layer': len(path)}

        for edge in outgoing.get(u, []):
            v = edge.bezier.node1
            if v not in visited:
                counter += 1
                heapq.heappush(heap, (d + edge.bezier.total_length(nodes), counter, v, path + [edge]))

    return result


def compute_all_routes(nodes, outgoing):
    """
    Run Dijkstra from every node. Returns:
        routes[src][dst] = {'path': [edge, ...], 'length': float, 'layer': int}
    Call this after rebuild_adjacency whenever the graph changes.
    """
    return {i: _dijkstra(i, nodes, outgoing) for i in range(len(nodes))}

# ── Logic ─────────────────────────────────────────────────────────────────────

def find_node_at(nodes, pos, radius=NODE_RADIUS):
    """Return the index of the topmost node under *pos*, or None."""
    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        if check_collision_point_circle(pos, Vector2(float(node[0]), float(node[1])), radius):
            return i
    return None


def update_node_mode(drag_state, drag_idx, nodes, edges, node_types, mouse_pos,
                     node_hit, selected_node):
    """
    Left-click empty space  → add node
    Left-drag a node        → move node (also selects it)
    Right-click a node      → delete node + its edges
    Edges and their handles are visible but not interactive.

    Returns (drag_state, drag_idx, selected_node).
    """
    mp       = np.array([mouse_pos.x, mouse_pos.y])
    pressed  = is_mouse_button_pressed( MouseButton.MOUSE_BUTTON_LEFT)
    released = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)
    rclick   = is_mouse_button_pressed( MouseButton.MOUSE_BUTTON_RIGHT)

    # Right-click → delete node and all edges that reference it
    if rclick and node_hit is not None:
        idx = node_hit
        edges[:] = [e for e in edges
                    if e.bezier.node0 != idx and e.bezier.node1 != idx]
        for e in edges:
            if e.bezier.node0 > idx: e.bezier.node0 -= 1
            if e.bezier.node1 > idx: e.bezier.node1 -= 1
        nodes.pop(idx)
        # Rebuild node_types with shifted indices
        new_types = {}
        for k, v in node_types.items():
            if k == idx:
                continue
            new_types[k - 1 if k > idx else k] = v
        node_types.clear()
        node_types.update(new_types)
        if selected_node == idx:
            selected_node = None
        elif selected_node is not None and selected_node > idx:
            selected_node -= 1
        return DragState.IDLE, None, selected_node

    match drag_state:
        case DragState.IDLE:
            if pressed:
                if node_hit is not None:
                    selected_node = node_hit
                    return DragState.DRAGGING, node_hit, selected_node
                else:
                    nodes.append(mp.copy())

        case DragState.DRAGGING:
            # Move the node continuously while dragging
            nodes[drag_idx] = mp.copy()
            if released:
                return DragState.IDLE, None, selected_node

    return drag_state, drag_idx, selected_node
