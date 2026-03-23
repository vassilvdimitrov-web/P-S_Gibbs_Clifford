import numpy as np
from scipy import integrate, optimize
import random

from pyray import *
import pickle

from dataclasses import dataclass,field
from enum import Enum, auto
from typing import List

import traffic_simulation as tsim
import entexi as eti
from metrics import TrafficMetrics

# ── Constants ────────────────────────────────────────────────────────────────

NODE_RADIUS   = 14
ARROW_SIZE    = 14
HANDLE_RADIUS = 8

GRAPH_W  = 900          # width of the graph/canvas area
PANEL_W  = 220          # width of the right-side properties panel
WINDOW_W = GRAPH_W + PANEL_W
WINDOW_H = 600

# UI strip at the top – graph interactions are blocked while the cursor is here
UI_HEIGHT = 64

# ── Bezier edge ───────────────────────────────────────────────────────────────

class QuadraticBezier:
    def __init__(self, node0, node1, nodes):
        self.node0 = node0
        self.node1 = node1
        self.ctrl  = (nodes[node0] + nodes[node1]) / 2.0
        self._dragging_ctrl = False
        self._r_prop_fn = None

    # ── control-point interaction ──────────────────────────────────────────

    def try_grab(self, mouse):
        if np.linalg.norm(self.ctrl - mouse) < HANDLE_RADIUS * 1.5:
            self._dragging_ctrl = True

    def drag(self, mouse):
        if self._dragging_ctrl:
            self.ctrl = mouse.copy()

    def release(self):
        self._dragging_ctrl = False

    @property
    def is_dragging(self):
        return self._dragging_ctrl

    # ── geometry ───────────────────────────────────────────────────────────

    def point_at(self, t, nodes):
        p0, p1 = nodes[self.node0], nodes[self.node1]
        return (1 - t)**2 * p0 + 2 * (1 - t) * t * self.ctrl + t**2 * p1

    def tangent_at(self, t, nodes):
        p0, p1 = nodes[self.node0], nodes[self.node1]
        return 2 * (1 - t) * (self.ctrl - p0) + 2 * t * (p1 - self.ctrl)

    def total_length(self, nodes):
        p0, p1 = nodes[self.node0], nodes[self.node1]
        return quadratic_bezier_arc_length(p0, self.ctrl, p1, 0, 1)[0]

    def total_length_from_to(self, t_start, t_end, nodes):
        p0, p1 = nodes[self.node0], nodes[self.node1]
        return quadratic_bezier_arc_length(p0, self.ctrl, p1, t_start, t_end)[0]

    def r_proportional(self, r_prop, nodes):
        p0, p1 = nodes[self.node0], nodes[self.node1]
        self._r_prop_fn = make_bezier_arc_length_solver(p0, self.ctrl, p1)[0]
        return self._r_prop_fn(r_prop)

    # ── drawing ────────────────────────────────────────────────────────────

    def draw(self, nodes, color=None):
        if color is None:
            color = BLACK

        steps = 100
        prev  = None
        tip   = None
        tip_t = None

        for i in range(steps + 1):
            t  = i / steps
            pt = self.point_at(t, nodes)

            if np.linalg.norm(pt - nodes[self.node0]) < NODE_RADIUS:
                prev = pt
                continue
            if np.linalg.norm(pt - nodes[self.node1]) < NODE_RADIUS:
                tip, tip_t = pt, t
                break

            if prev is not None:
                draw_line(int(prev[0]), int(prev[1]), int(pt[0]), int(pt[1]), color)
            prev = pt

        if tip is None:
            return

        tang   = self.tangent_at(tip_t, nodes)
        length = np.linalg.norm(tang)
        if length < 1e-6:
            return

        ux, uy       = tang / length
        tip_x, tip_y = tip
        base_x       = tip_x - ux * ARROW_SIZE
        base_y       = tip_y - uy * ARROW_SIZE
        px, py       = -uy, ux
        half         = ARROW_SIZE * 0.45

        draw_triangle(
            Vector2(tip_x, tip_y),
            Vector2(base_x - px * half, base_y - py * half),
            Vector2(base_x + px * half, base_y + py * half),
            color,
        )

    def draw_handle(self):
        draw_circle_lines(int(self.ctrl[0]), int(self.ctrl[1]), HANDLE_RADIUS, DARKGREEN)
        draw_circle(      int(self.ctrl[0]), int(self.ctrl[1]), HANDLE_RADIUS - 2, GREEN)

def quadratic_bezier_arc_length(P0, P1, P2, t_start, t_end, return_integrand=False):
    """
    Compute the arc length of a quadratic Bézier curve.

    Parameters:
        P0, P1, P2: array-like control points, e.g. [x, y]

    Returns:
        Arc length (float)
    """
    P0, P1, P2 = np.array(P0), np.array(P1), np.array(P2)

    A = P1 - P0  # vector A
    B = P2 - P1  # vector B

    A2 = np.dot(A, A)   # |A|²
    AB = np.dot(A, B)   # A·B
    B2 = np.dot(B, B)   # |B|²

    def integrand(t):
        return 2 * np.sqrt(A2*(1-t)**2 + 2*AB*t*(1-t) + B2*t**2)

    length, error = integrate.quad(integrand, t_start, t_end)
    if return_integrand:
        return length, error, integrand
    return length, error

def make_bezier_arc_length_solver(P0, P1, P2):
    """
    Returns a function that maps a fractional length [0, 1] to t [0, 1].
    """

    # Precompute total length
    total_length, _, integrand = quadratic_bezier_arc_length(P0, P1, P2, 0, 1, return_integrand=True)

    def t_at_fraction(fraction):
        """
        Given a fractional length in [0, 1], return the corresponding t.
        e.g. fraction=0.5 means halfway along the curve by arc length.
        """
        target = fraction * total_length

        # Solve: length(t) - target = 0
        def equation(t):
            length, _ = integrate.quad(integrand, 0, t)
            return length - target

        result = optimize.brentq(equation, 0, 1, xtol=1e-6)
        return result

    return t_at_fraction, total_length

# ── Enums ─────────────────────────────────────────────────────────────────────

class DragState(Enum):
    IDLE     = auto()
    DRAGGING = auto()

class EditMode(Enum):
    EDITNODES = auto()
    EDITEDGES = auto()

@dataclass
class EntryNode():
    spawn_probability: float = 0.05


@dataclass
class ExitNode():
    despawn_probability: float = 0.05


@dataclass
class Edge:
    bezier: QuadraticBezier
    node_type: EntryNode | ExitNode = field(default_factory=EntryNode)
    cars: List[eti.Car] = field(default_factory=list)
    def total_length(self, nodes):
        return self.bezier.total_length(nodes)  

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_node_at(nodes, pos, radius=NODE_RADIUS):
    """Return the index of the topmost node under *pos*, or None."""
    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        if check_collision_point_circle(pos, Vector2(float(node[0]), float(node[1])), radius):
            return i
    return None


def mouse_in_ui(mouse_pos):
    """True when the cursor is inside the top UI strip (graph area only)."""
    return mouse_pos.y < UI_HEIGHT and mouse_pos.x < GRAPH_W


def mouse_in_panel(mouse_pos):
    """True when the cursor is inside the right-side properties panel."""
    return mouse_pos.x >= GRAPH_W

# ── Mode updates ──────────────────────────────────────────────────────────────

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
                    new_idx = len(nodes) - 1
                    node_types[new_idx] = EntryNode()

        case DragState.DRAGGING:
            # Move the node continuously while dragging
            nodes[drag_idx] = mp.copy()
            if released:
                return DragState.IDLE, None, selected_node

    return drag_state, drag_idx, selected_node


def update_edge_mode(drag_state, drag_idx, nodes, edges, mouse_pos, node_hit):
    """
    Left-drag node → node    → create directed edge (if not duplicate)
    Drag green handle        → reshape edge curve
    Right-click green handle → delete edge
    Nodes are visible but cannot be moved or added.
    """
    mp       = np.array([mouse_pos.x, mouse_pos.y])
    pressed  = is_mouse_button_pressed( MouseButton.MOUSE_BUTTON_LEFT)
    released = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)
    rclick   = is_mouse_button_pressed( MouseButton.MOUSE_BUTTON_RIGHT)

    # Right-click near a handle → delete that edge
    if rclick:
        for i, edge in enumerate(edges):
            if np.linalg.norm(edge.bezier.ctrl - mp) < HANDLE_RADIUS * 2:
                edges.pop(i)
                return DragState.IDLE, None

    # Let each edge grab / update its control handle
    for edge in edges:
        if pressed:
            edge.bezier.try_grab(mp)
        edge.bezier.drag(mp)
        if released:
            edge.bezier.release()

    ctrl_grabbed = any(e.bezier.is_dragging for e in edges)

    match drag_state:
        case DragState.IDLE:
            if pressed and not ctrl_grabbed and node_hit is not None:
                return DragState.DRAGGING, node_hit

        case DragState.DRAGGING:
            if released:
                if (node_hit is not None
                        and node_hit != drag_idx
                        and not any(e.bezier.node0 == drag_idx and e.bezier.node1 == node_hit
                                    for e in edges)):
                    edges.append(Edge(QuadraticBezier(drag_idx, node_hit, nodes)))
                return DragState.IDLE, None

    return drag_state, drag_idx


# ── UI drawing ────────────────────────────────────────────────────────────────

def draw_button(rect, label, active):
    bg    = Color(30,  90, 200, 255) if active else Color(200, 200, 200, 255)
    fg    = WHITE                    if active else BLACK
    border= Color(20,  60, 140, 255) if active else Color(130, 130, 130, 255)
    draw_rectangle_rec(rect, bg)
    draw_rectangle_lines_ex(rect, 2, border)
    # Centre the text
    font_size = 18
    tw = measure_text(label, font_size)
    tx = int(rect.x + (rect.width  - tw) / 2)
    ty = int(rect.y + (rect.height - font_size) / 2)
    draw_text(label, tx, ty, font_size, fg)


def draw_ui(edit_mode, mouse_pos):
    """
    Draw the top toolbar and return the new EditMode if a button was clicked,
    otherwise return the current mode unchanged.
    """
    # Background strip (graph area only)
    draw_rectangle(0, 0, GRAPH_W, UI_HEIGHT, Color(240, 240, 245, 255))
    draw_line(0, UI_HEIGHT - 1, GRAPH_W, UI_HEIGHT - 1, Color(180, 180, 190, 255))

    btn_w, btn_h = 130, 34
    pad          = 10
    btn_y        = (UI_HEIGHT - btn_h) // 2

    btn_nodes = Rectangle(pad,           btn_y, btn_w, btn_h)
    btn_edges = Rectangle(pad + btn_w + 8, btn_y, btn_w, btn_h)

    draw_button(btn_nodes, "Edit Nodes", edit_mode == EditMode.EDITNODES)
    draw_button(btn_edges, "Edit Edges", edit_mode == EditMode.EDITEDGES)

    # Hint text
    if edit_mode == EditMode.EDITNODES:
        hint = "Click: add node   |   Drag: move node   |   Right-click: delete node"
    else:
        hint = "Drag: add edge   |   Drag handle: reshape   |   Right-click handle: delete edge"

    draw_text(hint, pad + btn_w * 2 + 24, btn_y + 8, 14, Color(80, 80, 100, 255))

    # Save / load hint (bottom-right of graph area)
    draw_text("S: save   L: load", GRAPH_W - 130, UI_HEIGHT - 18, 13, Color(140, 140, 160, 255))

    # Handle clicks only when released (avoids double-firing)
    new_mode = edit_mode
    if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
        if check_collision_point_rec(mouse_pos, btn_nodes):
            new_mode = EditMode.EDITNODES
        elif check_collision_point_rec(mouse_pos, btn_edges):
            new_mode = EditMode.EDITEDGES

    return new_mode


def _draw_float_editor(obj, attr, panel_x, y, mouse_pos):
    """Draw a  -  [value]  +  row for a float attribute on obj. Mutates obj."""
    val = getattr(obj, attr)

    btn_size  = 26
    val_box_w = 64
    gap       = 4

    minus_rect = Rectangle(panel_x + 10,                          y, btn_size, btn_size)
    val_rect   = Rectangle(panel_x + 10 + btn_size + gap,         y, val_box_w, btn_size)
    plus_rect  = Rectangle(panel_x + 10 + btn_size + gap + val_box_w + gap, y, btn_size, btn_size)

    # Minus button
    draw_button(minus_rect, "-", False)
    # Value box
    draw_rectangle_rec(val_rect, WHITE)
    draw_rectangle_lines_ex(val_rect, 1, Color(180, 180, 190, 255))
    val_str = f"{val:.3f}"
    tw = measure_text(val_str, 14)
    draw_text(val_str,
              int(val_rect.x + (val_box_w - tw) / 2),
              int(val_rect.y + (btn_size - 14) / 2),
              14, BLACK)
    # Plus button
    draw_button(plus_rect, "+", False)

    if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
        if check_collision_point_rec(mouse_pos, minus_rect):
            setattr(obj, attr, round(max(0.0, val - 0.01), 4))
        elif check_collision_point_rec(mouse_pos, plus_rect):
            setattr(obj, attr, round(min(1.0, val + 0.01), 4))


def draw_node_panel(selected_node, node_types, mouse_pos):
    """
    Draw the right-side node-properties panel.
    Handles type-selection clicks and property edits.
    Returns the (possibly mutated) node_types dict.
    """
    px = GRAPH_W

    # Panel background
    draw_rectangle(px, 0, PANEL_W, WINDOW_H, Color(245, 245, 250, 255))
    draw_line(px, 0, px, WINDOW_H, Color(180, 180, 190, 255))

    # Panel title bar
    draw_rectangle(px, 0, PANEL_W, UI_HEIGHT, Color(235, 235, 242, 255))
    draw_line(px, UI_HEIGHT - 1, px + PANEL_W, UI_HEIGHT - 1, Color(180, 180, 190, 255))
    title = "Node Properties"
    tw = measure_text(title, 16)
    draw_text(title, px + (PANEL_W - tw) // 2, (UI_HEIGHT - 16) // 2, 16,
              Color(40, 40, 60, 255))

    if selected_node is None:
        draw_text("No node selected", px + 14, UI_HEIGHT + 20, 13,
                  Color(130, 130, 150, 255))
        return node_types

    node_type = node_types.get(selected_node)

    y = UI_HEIGHT + 14

    # Node index label
    draw_text(f"Node {selected_node}", px + 14, y, 17, BLACK)
    y += 30

    # ── Type selector ──────────────────────────────────────────────────────
    draw_text("Type", px + 14, y, 13, Color(80, 80, 100, 255))
    y += 18

    type_defs = [
        ("None",  None),
        ("Entry", "entry"),
        ("Exit",  "exit"),
    ]
    btn_w = (PANEL_W - 28 - 2 * 6) // 3   # three equal buttons with gaps
    btn_h = 26

    for i, (label, _) in enumerate(type_defs):
        bx = px + 14 + i * (btn_w + 6)
        rect = Rectangle(bx, y, btn_w, btn_h)
        is_active = (
            (label == "None"  and node_type is None) or
            (label == "Entry" and isinstance(node_type, EntryNode)) or
            (label == "Exit"  and isinstance(node_type, ExitNode))
        )
        draw_button(rect, label, is_active)

        if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT) and \
                check_collision_point_rec(mouse_pos, rect):
            if label == "None":
                node_types.pop(selected_node, None)
            elif label == "Entry":
                if not isinstance(node_type, EntryNode):
                    node_types[selected_node] = EntryNode()
            else:  # Exit
                if not isinstance(node_type, ExitNode):
                    node_types[selected_node] = ExitNode()
            node_type = node_types.get(selected_node)

    y += btn_h + 18

    # ── Property editor ────────────────────────────────────────────────────
    if isinstance(node_type, EntryNode):
        draw_text("spawn probability", px + 14, y, 12, Color(80, 80, 100, 255))
        y += 16
        _draw_float_editor(node_type, "spawn_probability", px, y, mouse_pos)

    elif isinstance(node_type, ExitNode):
        draw_text("despawn probability", px + 14, y, 12, Color(80, 80, 100, 255))
        y += 16
        _draw_float_editor(node_type, "despawn_probability", px, y, mouse_pos)

    return node_types


# ── Main loop ─────────────────────────────────────────────────────────────────



def main():
    init_window(WINDOW_W, WINDOW_H, "")
    set_target_fps(60)

    nodes      = []
    edges      = []
    drag_state = DragState.IDLE
    drag_idx   = None
    edit_mode  = EditMode.EDITNODES
    
    #cars = [eti.Car(0, random.uniform(5, tsim.v_max), 5, 0.2, None), 
    #        eti.Car(200, random.uniform(5, tsim.v_max), 5, 0.2, None)]
    selected_node = None 
    node_types = {} 
    
    '''entry nodes'''
    entry_nodes = [] # Tracks the state (queues/lights) for each node
    cars = []        # Active cars driving on edges

    metrics = TrafficMetrics(road_length=1.0, v_max=tsim.v_max)

    while not window_should_close():
        # ── Input ──────────────────────────────────────────────────────────
        mouse_pos = get_mouse_position()
        mp        = np.array([mouse_pos.x, mouse_pos.y])
        in_ui     = mouse_in_ui(mouse_pos)
        in_panel  = mouse_in_panel(mouse_pos)

        # Only hit-test nodes when the cursor is in the graph area
        node_hit  = None if (in_ui or in_panel) else find_node_at(nodes, mouse_pos)

        # ── UI (mode buttons) ──────────────────────────────────────────────
        edit_mode = draw_ui(edit_mode, mouse_pos)   # also draws the strip

        # ── Graph logic (blocked while cursor is in UI strip or panel) ─────
        if not in_ui and not in_panel:
            if edit_mode == EditMode.EDITNODES:
                drag_state, drag_idx, selected_node = update_node_mode(
                    drag_state, drag_idx, nodes, edges, node_types,
                    mouse_pos, node_hit, selected_node)
            else:
                drag_state, drag_idx = update_edge_mode(
                    drag_state, drag_idx, nodes, edges, mouse_pos, node_hit)



        #only nodes marked as Entry generate cars and have queue/light logic.

        #collects the indices of all nodes that are currently marked as EntryNode
        entry_ids = [i for i, nt in node_types.items() if isinstance(nt, EntryNode)]
        #remove helper objects from entry_nodes if their node is no longer marked as Entry
        entry_nodes = [en for en in entry_nodes if en.node_id in entry_ids]             

        existing_ids = [en.node_id for en in entry_nodes]
        #for every node that should be Entry, check whether it already has an eti.EntryNode helper object.
        #ff not, add it to the list
        for node_id in entry_ids:
            if node_id not in existing_ids:
                entry_nodes.append(eti.EntryNode(node_id))



        # Car logic
        '''my car logic'''
        if nodes and edges:
        # A. Update Traffic Systems (Lights and Spawning into Queues)
            eti.update_traffic_lights(entry_nodes)
            eti.apply_traffic_lights(cars, entry_nodes, nodes)
            #node_indices = list(range(len(nodes))) 
            """changed"""
            eti.generate_entry_demand(entry_nodes, edges, len(nodes), probability=0.02)

            # B. Move cars from Node Queues onto the actual Edges
            eti.process_node_entries(entry_nodes, cars)

        # C. Physics Update
        tsim.update_velocities(cars, nodes, edges)
        tsim.update_positions(cars, nodes, edges)
        '''added nodes in this argument'''
        # D. Transitions (Handover from one edge to the next OR exiting)
        cars = eti.handle_edge_transitions(cars, edges, nodes)
        '''here I added another thing'''
        # This ensures if you clicked to add a node above,an entry_node is created for it immediately.
        


        # Metrics update
        total_road_length = 0.0
        for edge in edges:
            total_road_length += edge.bezier.total_length(nodes)        #total length should be the sum of all edge lengths

        metrics.road_length = total_road_length if total_road_length > 0 else 1.0
        metrics.compute(cars, entry_nodes)



        #we don't want this because it forces ALL nodes to behave like Entry nodes and it ignores the UI completely
        """
        if len(entry_nodes) < len(nodes):
            for i in range(len(entry_nodes), len(nodes)):
                entry_nodes.append(eti.EntryNode(i))
        elif len(entry_nodes) > len(nodes):
            entry_nodes = entry_nodes[:len(nodes)]
            for i, enode in enumerate(entry_nodes):
                enode.node_id = i
        """
        

        """ will comment out this"""
        """ Temporary (get rid of cars at end)
        tmp = []
        for car in cars:
            if not tsim.is_car_at_end_of_road(car):
                tmp += [car]
        
        cars = tmp
        """
        # Spawn in a new car if only 1 car is present
        ''''deleted this part
if len(cars) == 1:
    cars += [tsim.Car(0, random.uniform(5, tsim.v_max), 5, 0.2, None)]'''

        """will comment out this part too"""
        """  tsim.update_velocities(cars)
        tsim.update_positions(cars)
      """
        # ── Drawing ────────────────────────────────────────────────────────
        begin_drawing()
        clear_background(WHITE)

        draw_text(f"Cars: {len(cars)}", 20, 80, 20, RED)
        # Re-draw UI on top (begin_drawing clears)
        edit_mode = draw_ui(edit_mode, mouse_pos)

        # Edges
        for edge in edges:
            edge.bezier.draw(nodes)
            if edit_mode == EditMode.EDITEDGES:
                edge.bezier.draw_handle()

        # Edge-creation preview while dragging in edge mode
        if edit_mode == EditMode.EDITEDGES and drag_state == DragState.DRAGGING:
            preview_nodes = nodes + [mp]
            preview = QuadraticBezier(drag_idx, len(nodes), preview_nodes)
            preview.draw(preview_nodes, GRAY)

        # Nodes
        for i, node in enumerate(nodes):
            nt = node_types.get(i)

            if i == drag_idx and drag_state == DragState.DRAGGING:
                col = ORANGE
            elif i == node_hit:
                col = SKYBLUE
            elif isinstance(nt, EntryNode):
                col = Color(60, 180, 60, 255)    # green = entry
            elif isinstance(nt, ExitNode):
                col = Color(210, 60, 60, 255)    # red   = exit
            else:
                col = DARKBLUE if edit_mode == EditMode.EDITNODES else BLUE

            draw_circle(int(node[0]), int(node[1]), NODE_RADIUS, col)
            draw_circle_lines(int(node[0]), int(node[1]), NODE_RADIUS, BLACK)

            # Selection ring
            if i == selected_node and edit_mode == EditMode.EDITNODES:
                draw_circle_lines(int(node[0]), int(node[1]), NODE_RADIUS + 4,
                                  Color(255, 200, 0, 255))

        # Cursor dot (graph area only)
        if not in_ui and not in_panel:
            draw_circle(int(mp[0]), int(mp[1]), 4, RED)

        # Right-side panel
        draw_node_panel(selected_node, node_types, mouse_pos)

        # ── Save / Load ────────────────────────────────────────────────────
        if is_key_pressed(KeyboardKey.KEY_S):
            pickle.dump({"nodes": nodes, "edges": edges, "node_types": node_types},
                        open("data.pkl", "wb"))

        if is_key_pressed(KeyboardKey.KEY_L):
            data  = pickle.load(open("data.pkl", "rb"))
            nodes = data["nodes"]
            edges = data["edges"]
            node_types = data.get("node_types", {})
            drag_state = DragState.IDLE
            drag_idx   = None
            selected_node = None


        # ── Car Drawing ────────────────────────────────────────────────────
        if edges:
            for car in cars:
                # Use the length of the specific edge the car is currently on
                edge_len = car.current_edge.bezier.total_length(nodes)
                if edge_len > 0:
                    relative_pos = car.x / edge_len
                    # Ensure we don't go out of bounds before the transition logic catches it
                    relative_pos = max(0, min(1, relative_pos))
                    
                    t = car.current_edge.bezier.r_proportional(relative_pos, nodes)
                    pos = car.current_edge.bezier.point_at(t, nodes)
                    
                    # Draw car - different color for active cars
                    #draw_circle(int(pos[0]), int(pos[1]), 5, RED)
                    car_color = PURPLE if getattr(car, "is_bad_driver", False) else RED
                    draw_circle(int(pos[0]), int(pos[1]), 5, car_color)

            # --- Draw Queue counts above nodes ---
            for node in entry_nodes:
                pos = nodes[node.node_id]
    
                # 2. Determine the color based on the entry light state
                light_color = GREEN if node.entry_green else RED
                
                # 3. Now draw the light (using the 'pos' we just defined)
                draw_circle(int(pos[0]) + 20, int(pos[1]), 5, light_color)
                
                # 4. Draw the queue count if anyone is waiting
                if node.waiting_queue:
                    draw_text(str(len(node.waiting_queue)), int(pos[0]) - 5, int(pos[1]) - 25, 12, DARKGRAY)

        end_drawing()
    metrics.save("traffic_data.csv")
    close_window()


if __name__ == "__main__":
    main()
