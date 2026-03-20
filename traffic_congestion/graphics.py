import numpy as np
from pyray import *
from enum import Enum, auto
import pickle
from scipy import integrate, optimize
import traffic_simulation as tsim
import random

import entexi as ex
# ── Constants ────────────────────────────────────────────────────────────────

NODE_RADIUS   = 14
ARROW_SIZE    = 14
HANDLE_RADIUS = 8

WINDOW_W = 900
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_node_at(nodes, pos, radius=NODE_RADIUS):
    """Return the index of the topmost node under *pos*, or None."""
    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        if check_collision_point_circle(pos, Vector2(float(node[0]), float(node[1])), radius):
            return i
    return None


def mouse_in_ui(mouse_pos):
    """True when the cursor is inside the top UI strip."""
    return mouse_pos.y < UI_HEIGHT

# ── Mode updates ──────────────────────────────────────────────────────────────

def update_node_mode(drag_state, drag_idx, nodes, edges, mouse_pos, node_hit):
    """
    Left-click empty space  → add node
    Left-drag a node        → move node
    Right-click a node      → delete node + its edges
    Edges and their handles are visible but not interactive.
    """
    mp       = np.array([mouse_pos.x, mouse_pos.y])
    pressed  = is_mouse_button_pressed( MouseButton.MOUSE_BUTTON_LEFT)
    released = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)
    rclick   = is_mouse_button_pressed( MouseButton.MOUSE_BUTTON_RIGHT)

    # Right-click → delete node and all edges that reference it
    if rclick and node_hit is not None:
        idx = node_hit
        edges[:] = [e for e in edges if e.node0 != idx and e.node1 != idx]
        # Fix indices in remaining edges
        for e in edges:
            if e.node0 > idx: e.node0 -= 1
            if e.node1 > idx: e.node1 -= 1
        nodes.pop(idx)
        return DragState.IDLE, None

    match drag_state:
        case DragState.IDLE:
            if pressed:
                if node_hit is not None:
                    return DragState.DRAGGING, node_hit
                else:
                    nodes.append(mp.copy())

        case DragState.DRAGGING:
            # Move the node continuously while dragging
            nodes[drag_idx] = mp.copy()
            if released:
                return DragState.IDLE, None

    return drag_state, drag_idx


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
            if np.linalg.norm(edge.ctrl - mp) < HANDLE_RADIUS * 2:
                edges.pop(i)
                return DragState.IDLE, None

    # Let each edge grab / update its control handle
    for edge in edges:
        if pressed:
            edge.try_grab(mp)
        edge.drag(mp)
        if released:
            edge.release()

    ctrl_grabbed = any(e.is_dragging for e in edges)

    match drag_state:
        case DragState.IDLE:
            if pressed and not ctrl_grabbed and node_hit is not None:
                return DragState.DRAGGING, node_hit

        case DragState.DRAGGING:
            if released:
                if (node_hit is not None
                        and node_hit != drag_idx
                        and not any(e.node0 == drag_idx and e.node1 == node_hit
                                    for e in edges)):
                    edges.append(QuadraticBezier(drag_idx, node_hit, nodes))
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
    # Background strip
    draw_rectangle(0, 0, WINDOW_W, UI_HEIGHT, Color(240, 240, 245, 255))
    draw_line(0, UI_HEIGHT - 1, WINDOW_W, UI_HEIGHT - 1, Color(180, 180, 190, 255))

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

    # Save / load hint (bottom-right)
    draw_text("S: save   L: load", WINDOW_W - 130, UI_HEIGHT - 18, 13, Color(140, 140, 160, 255))

    # Handle clicks only when released (avoids double-firing)
    new_mode = edit_mode
    if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
        if check_collision_point_rec(mouse_pos, btn_nodes):
            new_mode = EditMode.EDITNODES
        elif check_collision_point_rec(mouse_pos, btn_edges):
            new_mode = EditMode.EDITEDGES

    return new_mode


# ── Main loop ─────────────────────────────────────────────────────────────────



def main():
    init_window(WINDOW_W, WINDOW_H, "")
    set_target_fps(60)

    nodes      = []
    edges      = []
    drag_state = DragState.IDLE
    drag_idx   = None
    edit_mode  = EditMode.EDITNODES
    cars = [tsim.Car(0, random.uniform(5, tsim.v_max), 5, 0.2, None), 
            tsim.Car(200, random.uniform(5, tsim.v_max), 5, 0.2, None)]


    while not window_should_close():
        # ── Input ──────────────────────────────────────────────────────────
        mouse_pos = get_mouse_position()
        mp        = np.array([mouse_pos.x, mouse_pos.y])
        in_ui     = mouse_in_ui(mouse_pos)

        # Only hit-test nodes when the cursor is in the graph area
        node_hit  = None if in_ui else find_node_at(nodes, mouse_pos)

        # ── UI (mode buttons) ──────────────────────────────────────────────
        edit_mode = draw_ui(edit_mode, mouse_pos)   # also draws the strip

        # ── Graph logic (blocked while cursor is in UI strip) ──────────────
        if not in_ui:
            if edit_mode == EditMode.EDITNODES:
                drag_state, drag_idx = update_node_mode(
                    drag_state, drag_idx, nodes, edges, mouse_pos, node_hit)
            else:
                drag_state, drag_idx = update_edge_mode(
                    drag_state, drag_idx, nodes, edges, mouse_pos, node_hit)

        # Car logic

        #Temporary (get rid of cars at end)
        tmp = []
        for car in cars:
            if not tsim.is_car_at_end_of_road(car):
                tmp += [car]
        
        cars = tmp

        # Spawn in a new car if only 1 car is present
        if len(cars) == 1:
            cars += [tsim.Car(0, random.uniform(5, tsim.v_max), 5, 0.2, None)]

        tsim.update_velocities(cars)
        tsim.update_positions(cars)

        # ── Drawing ────────────────────────────────────────────────────────
        begin_drawing()
        clear_background(WHITE)

        # Re-draw UI on top (begin_drawing clears)
        edit_mode = draw_ui(edit_mode, mouse_pos)

        # Edges
        for edge in edges:
            edge.draw(nodes)
            if edit_mode == EditMode.EDITEDGES:
                edge.draw_handle()

        # Edge-creation preview while dragging in edge mode
        if edit_mode == EditMode.EDITEDGES and drag_state == DragState.DRAGGING:
            preview_nodes = nodes + [mp]
            preview = QuadraticBezier(drag_idx, len(nodes), preview_nodes)
            preview.draw(preview_nodes, GRAY)

        # Nodes
        for i, node in enumerate(nodes):
            if i == drag_idx and drag_state == DragState.DRAGGING:
                col = ORANGE          # currently being moved
            elif i == node_hit:
                col = SKYBLUE         # hovered
            else:
                col = DARKBLUE if edit_mode == EditMode.EDITNODES else BLUE
            draw_circle(int(node[0]), int(node[1]), NODE_RADIUS, col)
            draw_circle_lines(int(node[0]), int(node[1]), NODE_RADIUS, BLACK)

        # Cursor dot (graph area only)
        if not in_ui:
            draw_circle(int(mp[0]), int(mp[1]), 4, RED)

        # ── Save / Load ────────────────────────────────────────────────────
        if is_key_pressed(KeyboardKey.KEY_S):
            pickle.dump({"nodes": nodes, "edges": edges}, open("data.pkl", "wb"))

        if is_key_pressed(KeyboardKey.KEY_L):
            data  = pickle.load(open("data.pkl", "rb"))
            nodes = data["nodes"]
            edges = data["edges"]
            drag_state = DragState.IDLE
            drag_idx   = None


        # ── Car Drawing ────────────────────────────────────────────────────
        if edges:
            for car in cars:
                t = edges[0].r_proportional(tsim.car_relative_position(car), nodes)
                pos = edges[0].point_at(t, nodes)
                draw_circle(int(pos[0]), int(pos[1]), 4, Color(180, 180, 255, 255))

        end_drawing()

    close_window()


if __name__ == "__main__":
    main()