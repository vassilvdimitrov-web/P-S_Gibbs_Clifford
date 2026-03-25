import numpy as np
from scipy import integrate, optimize
import random
import pickle

from pyray import *

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


#make the simulation deterministic
random.seed(0)
SAVED_FILENAME = "./results/ring8_e3x3_sp0.06_bd0.25_sd20_vm70.pkl"

# ── traffic_simulation ────────────────────────────────────────────────────────


safe_distance = 10
dt = 0.01

v_max = 60.0
bad_driver_safe_distance_factor = 5
bad_driver_acceleartion_factor = 0.7
bad_driver_safe_brake_factor = 7.5
BAD_DRIVER_FRACTION = 0.50

default_acceleration = 3.0
default_brake_factor = 5.0


def local_speed_limit(x, edge=None):
    if edge is not None and hasattr(edge, "speed_limit"):
        return float(edge.speed_limit)
    return v_max


def cars_on_edge(cars, edge):
    result = []
    for car in cars:
        if car.current_edge is edge and not car.is_waiting_to_enter:
            result.append(car)
    return result


def next_car_ahead(car, edge_cars):
    ahead = []
    for other in edge_cars:
        if other is not car and other.x > car.x:
            ahead.append(other)

    if len(ahead) == 0:
        return None

    leader = ahead[0]
    for other in ahead[1:]:
        if other.x < leader.x:
            leader = other
    return leader


def gap_to_next_car(car, leader):
    gap = leader.x - car.x - leader.l
    if gap < 0.0:
        return 0.0
    return gap


def edge_length(edge, nodes):
    return float(edge.bezier.total_length(nodes))


def update_velocities_on_edge(edge, cars, nodes):
    edge_cars = cars_on_edge(cars, edge)
    edge_cars.sort(key=lambda c: c.x)

    for car in edge_cars:
        if getattr(car, "is_bad_driver", False):
            safe_dist = safe_distance * bad_driver_safe_distance_factor
            accel = default_acceleration * bad_driver_acceleartion_factor
            brake = default_brake_factor * bad_driver_safe_brake_factor
        else:
            safe_dist = safe_distance
            accel = default_acceleration
            brake = default_brake_factor

        leader = next_car_ahead(car, edge_cars)
        limit = local_speed_limit(car.x, edge)

        if leader is None:
            car.v = min(limit, car.v + accel)
        else:
            gap = gap_to_next_car(car, leader)

            if gap < safe_dist:
                car.v = max(0.0, car.v - brake * car.reac)
            else:
                car.v = min(limit, car.v + accel)


def update_positions_on_edge(edge, cars, nodes):
    length = edge_length(edge, nodes)
    for car in cars:
        if car.current_edge is not edge:
            continue
        if car.is_waiting_to_enter:
            continue

        car.x += car.v * dt

        if car.x > length:
            car.x = length


def update_velocities(cars, nodes, edges=None):
    if edges is None:
        edges = []
        for car in cars:
            if not car.is_waiting_to_enter and car.current_edge is not None:
                if car.current_edge not in edges:
                    edges.append(car.current_edge)

    for edge in edges:
        update_velocities_on_edge(edge=edge, cars=cars, nodes=nodes)


def update_positions(cars, nodes, edges=None):
    if edges is None:
        edges = []
        for car in cars:
            if not car.is_waiting_to_enter and car.current_edge is not None:
                if car.current_edge not in edges:
                    edges.append(car.current_edge)

    for edge in edges:
        update_positions_on_edge(edge=edge, cars=cars, nodes=nodes)


# ── entexi ────────────────────────────────────────────────────────────────────

class EntryNodeHelper:
    def __init__(self, node_id):
        self.node_id = node_id
        self.waiting_queue = []
        self.entry_green = False
        self.main_green = True
        self.timer = 0
        self.phase_duration = random.randint(150, 300)


class Car:
    def __init__(self, current_edge, velocity, reaction_speed, destination_node_idx, length=4.5, is_bad_driver=False):
        self.current_edge = current_edge
        self.x = 0.0
        self.v = velocity
        self.reac = reaction_speed
        self.l = length
        self.destination_node_idx = destination_node_idx
        self.is_bad_driver = is_bad_driver
        self.is_waiting_to_enter = True


def can_safely_enter(target_edge, active_cars, safe_distance=0.5):
    for car in active_cars:
        if car.current_edge == target_edge:
            if car.x < safe_distance:
                return False
    return True


def process_node_entries(entry_nodes, active_cars):
    for node in entry_nodes:
        if len(node.waiting_queue) > 0 and node.entry_green:
            next_car = node.waiting_queue[0]
            if can_safely_enter(next_car.current_edge, active_cars, safe_distance=25.0):
                entering_car = node.waiting_queue.pop(0)
                entering_car.is_waiting_to_enter = False
                entering_car.x = 0.0
                active_cars.append(entering_car)


def handle_edge_transitions(cars, edges, nodes):
    remaining_cars = []
    for car in cars:
        edge_len = car.current_edge.bezier.total_length(nodes)

        if car.x >= edge_len:
            if car.current_edge.bezier.node1 == car.destination_node_idx:
                continue

            next_options = [e for e in edges if e.bezier.node0 == car.current_edge.bezier.node1]

            if next_options:
                car.current_edge = random.choice(next_options)
                car.x = 0
                remaining_cars.append(car)
        else:
            remaining_cars.append(car)
    return remaining_cars


def generate_entry_demand(entry_nodes, edges, node_types, probability=0.05):
    for node in entry_nodes:
        if random.random() < probability:
            outgoing = [e for e in edges if e.bezier.node0 == node.node_id]
            if outgoing:
                start_edge = random.choice(outgoing)
                exit_nodes = []
                weights = []

                for i, t in node_types.items():
                    if t.__class__.__name__ == "ExitNode":
                        exit_nodes.append(i)
                        weights.append(max(0.0001, t.demand))

                if not exit_nodes:
                    continue

                dest = random.choices(exit_nodes, weights=weights, k=1)[0]
                is_bad = random.random() < BAD_DRIVER_FRACTION
                new_car = Car(
                    start_edge,
                    random.uniform(2, 5),
                    random.uniform(0.5, 1.5),
                    dest,
                    is_bad_driver=is_bad,
                )
                node.waiting_queue.append(new_car)


def update_traffic_lights(entry_nodes):
    for node in entry_nodes:
        node.timer += 1
        if node.timer > node.phase_duration:
            node.entry_green = not node.entry_green
            node.main_green = not node.main_green
            node.timer = 0


def apply_traffic_lights(cars, entry_nodes, nodes):
    for car in cars:
        target_node_id = car.current_edge.bezier.node1
        for enode in entry_nodes:
            if enode.node_id == target_node_id:
                edge_len = car.current_edge.bezier.total_length(nodes)
                dist_to_light = edge_len - car.x

                if dist_to_light < 40:
                    if not enode.main_green:
                        car.v = 0


# ── metrics ───────────────────────────────────────────────────────────────────

class TrafficMetrics:
    def __init__(self, road_length, v_max):
        self.road_length = road_length
        self.v_max = v_max
        self.history = []

    def compute(self, cars, entry_nodes):
        if not cars:
            density = 0
            avg_v = 0
            flow = 0
        else:
            density = len(cars) / self.road_length
            avg_v = sum(car.v for car in cars) / len(cars)
            flow = density * avg_v

        if entry_nodes:
            avg_queue = sum(len(n.waiting_queue) for n in entry_nodes) / len(entry_nodes)
            max_queue = max(len(n.waiting_queue) for n in entry_nodes)
        else:
            avg_queue = 0
            max_queue = 0

        congested = False
        if cars:
            if avg_v < 0.7 * self.v_max:
                congested = True
        if avg_queue > 20 or max_queue > 30:
            congested = True

        self.history.append({
            "density": density,
            "velocity": avg_v,
            "flow": flow,
            "congested": congested,
            "cars": len(cars),
            "queue": avg_queue,
            "max_queue": max_queue,
        })

    def save(self, filename="traffic_data.npz"):
        dtype = np.dtype([
            ("density",   np.float64),
            ("velocity",  np.float64),
            ("flow",      np.float64),
            ("congested", np.bool_),
            ("cars",      np.int64),
            ("queue",     np.float64),
            ("max_queue", np.int64),
        ])
        arr = np.array(
            [(h["density"], h["velocity"], h["flow"], h["congested"],
              h["cars"], h["queue"], h["max_queue"])
             for h in self.history],
            dtype=dtype,
        )
        np.savez_compressed(filename, traffic=arr)

    @staticmethod
    def load(filename="traffic_data.npz"):
        arr = np.load(filename)
        return arr

    def first_congestion_step(self):
        for i, h in enumerate(self.history):
            if h["congested"]:
                return i
        return None


# ── Constants ────────────────────────────────────────────────────────────────

NODE_RADIUS   = 14
ARROW_SIZE    = 14
HANDLE_RADIUS = 8

GRAPH_W  = 900
PANEL_W  = 220
WINDOW_W = GRAPH_W + PANEL_W
WINDOW_H = 600

UI_HEIGHT = 64

# ── Bezier edge ───────────────────────────────────────────────────────────────

class QuadraticBezier:
    def __init__(self, node0, node1, nodes):
        self.node0 = node0
        self.node1 = node1
        self.ctrl  = (nodes[node0] + nodes[node1]) / 2.0
        self._dragging_ctrl = False
        self._r_prop_fn = None

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
    P0, P1, P2 = np.array(P0), np.array(P1), np.array(P2)

    A = P1 - P0
    B = P2 - P1

    A2 = np.dot(A, A)
    AB = np.dot(A, B)
    B2 = np.dot(B, B)

    def integrand(t):
        return 2 * np.sqrt(A2*(1-t)**2 + 2*AB*t*(1-t) + B2*t**2)

    length, error = integrate.quad(integrand, t_start, t_end)
    if return_integrand:
        return length, error, integrand
    return length, error


def make_bezier_arc_length_solver(P0, P1, P2):
    total_length, _, integrand = quadratic_bezier_arc_length(P0, P1, P2, 0, 1, return_integrand=True)

    def t_at_fraction(fraction):
        target = fraction * total_length

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
    demand: float = 1.0


@dataclass
class Edge:
    bezier: QuadraticBezier
    node_type: EntryNode | ExitNode = field(default_factory=EntryNode)
    cars: List[Car] = field(default_factory=list)

    def total_length(self, nodes):
        return self.bezier.total_length(nodes)


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_node_at(nodes, pos, radius=NODE_RADIUS):
    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        if check_collision_point_circle(pos, Vector2(float(node[0]), float(node[1])), radius):
            return i
    return None


def mouse_in_ui(mouse_pos):
    return mouse_pos.y < UI_HEIGHT and mouse_pos.x < GRAPH_W


def mouse_in_panel(mouse_pos):
    return mouse_pos.x >= GRAPH_W


# ── Mode updates ──────────────────────────────────────────────────────────────

def update_node_mode(drag_state, drag_idx, nodes, edges, node_types, mouse_pos,
                     node_hit, selected_node):
    mp       = np.array([mouse_pos.x, mouse_pos.y])
    pressed  = is_mouse_button_pressed( MouseButton.MOUSE_BUTTON_LEFT)
    released = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)
    rclick   = is_mouse_button_pressed( MouseButton.MOUSE_BUTTON_RIGHT)

    if rclick and node_hit is not None:
        idx = node_hit
        edges[:] = [e for e in edges
                    if e.bezier.node0 != idx and e.bezier.node1 != idx]
        for e in edges:
            if e.bezier.node0 > idx: e.bezier.node0 -= 1
            if e.bezier.node1 > idx: e.bezier.node1 -= 1
        nodes.pop(idx)
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
            nodes[drag_idx] = mp.copy()
            if released:
                return DragState.IDLE, None, selected_node

    return drag_state, drag_idx, selected_node


def update_edge_mode(drag_state, drag_idx, nodes, edges, mouse_pos, node_hit):
    mp       = np.array([mouse_pos.x, mouse_pos.y])
    pressed  = is_mouse_button_pressed( MouseButton.MOUSE_BUTTON_LEFT)
    released = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)
    rclick   = is_mouse_button_pressed( MouseButton.MOUSE_BUTTON_RIGHT)

    if rclick:
        for i, edge in enumerate(edges):
            if np.linalg.norm(edge.bezier.ctrl - mp) < HANDLE_RADIUS * 2:
                edges.pop(i)
                return DragState.IDLE, None

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
    bg     = Color(30,  90, 200, 255) if active else Color(200, 200, 200, 255)
    fg     = WHITE                    if active else BLACK
    border = Color(20,  60, 140, 255) if active else Color(130, 130, 130, 255)
    draw_rectangle_rec(rect, bg)
    draw_rectangle_lines_ex(rect, 2, border)
    font_size = 18
    tw = measure_text(label, font_size)
    tx = int(rect.x + (rect.width  - tw) / 2)
    ty = int(rect.y + (rect.height - font_size) / 2)
    draw_text(label, tx, ty, font_size, fg)


def draw_ui(edit_mode, mouse_pos):
    draw_rectangle(0, 0, GRAPH_W, UI_HEIGHT, Color(240, 240, 245, 255))
    draw_line(0, UI_HEIGHT - 1, GRAPH_W, UI_HEIGHT - 1, Color(180, 180, 190, 255))

    btn_w, btn_h = 130, 34
    pad          = 10
    btn_y        = (UI_HEIGHT - btn_h) // 2

    btn_nodes = Rectangle(pad,            btn_y, btn_w, btn_h)
    btn_edges = Rectangle(pad + btn_w + 8, btn_y, btn_w, btn_h)

    draw_button(btn_nodes, "Edit Nodes", edit_mode == EditMode.EDITNODES)
    draw_button(btn_edges, "Edit Edges", edit_mode == EditMode.EDITEDGES)

    if edit_mode == EditMode.EDITNODES:
        hint = "Click: add node   |   Drag: move node   |   Right-click: delete node"
    else:
        hint = "Drag: add edge   |   Drag handle: reshape   |   Right-click handle: delete edge"

    draw_text(hint, pad + btn_w * 2 + 24, btn_y + 8, 14, Color(80, 80, 100, 255))
    draw_text("S: save   L: load", GRAPH_W - 130, UI_HEIGHT - 18, 13, Color(140, 140, 160, 255))

    new_mode = edit_mode
    if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
        if check_collision_point_rec(mouse_pos, btn_nodes):
            new_mode = EditMode.EDITNODES
        elif check_collision_point_rec(mouse_pos, btn_edges):
            new_mode = EditMode.EDITEDGES

    return new_mode


def _draw_float_editor(obj, attr, panel_x, y, mouse_pos):
    val = getattr(obj, attr)

    btn_size  = 26
    val_box_w = 64
    gap       = 4

    minus_rect = Rectangle(panel_x + 10,                                    y, btn_size, btn_size)
    val_rect   = Rectangle(panel_x + 10 + btn_size + gap,                   y, val_box_w, btn_size)
    plus_rect  = Rectangle(panel_x + 10 + btn_size + gap + val_box_w + gap, y, btn_size, btn_size)

    draw_button(minus_rect, "-", False)
    draw_rectangle_rec(val_rect, WHITE)
    draw_rectangle_lines_ex(val_rect, 1, Color(180, 180, 190, 255))
    val_str = f"{val:.3f}"
    tw = measure_text(val_str, 14)
    draw_text(val_str,
              int(val_rect.x + (val_box_w - tw) / 2),
              int(val_rect.y + (btn_size - 14) / 2),
              14, BLACK)
    draw_button(plus_rect, "+", False)

    if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
        if check_collision_point_rec(mouse_pos, minus_rect):
            setattr(obj, attr, round(max(0.0, val - 0.01), 4))
        elif check_collision_point_rec(mouse_pos, plus_rect):
            setattr(obj, attr, round(min(1.0, val + 0.01), 4))


def draw_node_panel(selected_node, node_types, mouse_pos):
    px = GRAPH_W

    draw_rectangle(px, 0, PANEL_W, WINDOW_H, Color(245, 245, 250, 255))
    draw_line(px, 0, px, WINDOW_H, Color(180, 180, 190, 255))

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

    draw_text(f"Node {selected_node}", px + 14, y, 17, BLACK)
    y += 30

    draw_text("Type", px + 14, y, 13, Color(80, 80, 100, 255))
    y += 18

    type_defs = [
        ("None",  None),
        ("Entry", "entry"),
        ("Exit",  "exit"),
    ]
    btn_w = (PANEL_W - 28 - 2 * 6) // 3
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

    if isinstance(node_type, EntryNode):
        draw_text("spawn probability", px + 14, y, 12, Color(80, 80, 100, 255))
        y += 16
        _draw_float_editor(node_type, "spawn_probability", px, y, mouse_pos)

    elif isinstance(node_type, ExitNode):
        draw_text("despawn probability", px + 14, y, 12, Color(80, 80, 100, 255))
        y += 16
        _draw_float_editor(node_type, "despawn_probability", px, y, mouse_pos)
        y += 40
        draw_text("demand", px + 14, y, 12, Color(80, 80, 100, 255))
        y += 16
        _draw_float_editor(node_type, "demand", px, y, mouse_pos)
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

    selected_node = None
    node_types = {}

    entry_nodes = []
    cars = []

    metrics = TrafficMetrics(road_length=1.0, v_max=v_max)

    while not window_should_close():
        # ── Input ──────────────────────────────────────────────────────────
        mouse_pos = get_mouse_position()
        mp        = np.array([mouse_pos.x, mouse_pos.y])
        in_ui     = mouse_in_ui(mouse_pos)
        in_panel  = mouse_in_panel(mouse_pos)

        node_hit  = None if (in_ui or in_panel) else find_node_at(nodes, mouse_pos)

        # ── UI (mode buttons) ──────────────────────────────────────────────
        edit_mode = draw_ui(edit_mode, mouse_pos)

        # ── Graph logic ────────────────────────────────────────────────────
        if not in_ui and not in_panel:
            if edit_mode == EditMode.EDITNODES:
                drag_state, drag_idx, selected_node = update_node_mode(
                    drag_state, drag_idx, nodes, edges, node_types,
                    mouse_pos, node_hit, selected_node)
            else:
                drag_state, drag_idx = update_edge_mode(
                    drag_state, drag_idx, nodes, edges, mouse_pos, node_hit)

        entry_ids = [i for i, nt in node_types.items() if isinstance(nt, EntryNode)]
        entry_nodes = [en for en in entry_nodes if en.node_id in entry_ids]

        existing_ids = [en.node_id for en in entry_nodes]
        for node_id in entry_ids:
            if node_id not in existing_ids:
                entry_nodes.append(EntryNodeHelper(node_id))

        # Car logic
        if nodes and edges:
            update_traffic_lights(entry_nodes)
            apply_traffic_lights(cars, entry_nodes, nodes)
            generate_entry_demand(entry_nodes, edges, node_types, probability=0.02)
            process_node_entries(entry_nodes, cars)

        update_velocities(cars, nodes, edges)
        update_positions(cars, nodes, edges)
        cars = handle_edge_transitions(cars, edges, nodes)

        # Metrics update
        total_road_length = 0.0
        for edge in edges:
            total_road_length += edge.bezier.total_length(nodes)

        metrics.road_length = total_road_length if total_road_length > 0 else 1.0
        metrics.compute(cars, entry_nodes)

        # ── Drawing ────────────────────────────────────────────────────────
        begin_drawing()
        clear_background(WHITE)

        draw_text(f"Cars: {len(cars)}", 20, 80, 20, RED)
        edit_mode = draw_ui(edit_mode, mouse_pos)

        # Edges
        for edge in edges:
            edge.bezier.draw(nodes)
            if edit_mode == EditMode.EDITEDGES:
                edge.bezier.draw_handle()

        # Edge-creation preview
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
                col = Color(60, 180, 60, 255)
            elif isinstance(nt, ExitNode):
                col = Color(210, 60, 60, 255)
            else:
                col = DARKBLUE if edit_mode == EditMode.EDITNODES else BLUE

            draw_circle(int(node[0]), int(node[1]), NODE_RADIUS, col)
            draw_circle_lines(int(node[0]), int(node[1]), NODE_RADIUS, BLACK)

            if i == selected_node and edit_mode == EditMode.EDITNODES:
                draw_circle_lines(int(node[0]), int(node[1]), NODE_RADIUS + 4,
                                  Color(255, 200, 0, 255))

        # Cursor dot
        if not in_ui and not in_panel:
            draw_circle(int(mp[0]), int(mp[1]), 4, RED)

        # Right-side panel
        draw_node_panel(selected_node, node_types, mouse_pos)

        # ── Save / Load ────────────────────────────────────────────────────
        if is_key_pressed(KeyboardKey.KEY_S):
            with open(SAVED_FILENAME, "wb") as f:
                pickle.dump({"nodes": nodes, "edges": edges, "node_types": node_types}, f)

        if is_key_pressed(KeyboardKey.KEY_L):
            try:
                with open(SAVED_FILENAME, "rb") as f:
                    data = pickle.load(f)
                nodes = data["nodes"]
                edges = data["edges"]
                raw_types = data.get("node_types", {})
                node_types = {}
                for k, v in raw_types.items():
                    if isinstance(v, dict):
                        if v.get("__type__") == "EntryNode":
                            node_types[k] = EntryNode(spawn_probability=v.get("spawn_probability", 0.05))
                        elif v.get("__type__") == "ExitNode":
                            node_types[k] = ExitNode(despawn_probability=v.get("despawn_probability", 0.05),
                                                      demand=v.get("demand", 1.0))
                    else:
                        node_types[k] = v
                drag_state = DragState.IDLE
                drag_idx   = None
                selected_node = None
            except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
                print(f"Load failed: {e}")

        # ── Car Drawing ────────────────────────────────────────────────────
        if edges:
            for car in cars:
                edge_len = car.current_edge.bezier.total_length(nodes)
                if edge_len > 0:
                    relative_pos = car.x / edge_len
                    relative_pos = max(0, min(1, relative_pos))

                    t = car.current_edge.bezier.r_proportional(relative_pos, nodes)
                    pos = car.current_edge.bezier.point_at(t, nodes)

                    car_color = PURPLE if getattr(car, "is_bad_driver", False) else RED
                    draw_circle(int(pos[0]), int(pos[1]), 5, car_color)

            for node in entry_nodes:
                pos = nodes[node.node_id]

                light_color = GREEN if node.entry_green else RED
                draw_circle(int(pos[0]) + 20, int(pos[1]), 5, light_color)

                if node.waiting_queue:
                    draw_text(str(len(node.waiting_queue)), int(pos[0]) - 5, int(pos[1]) - 25, 12, DARKGRAY)

        end_drawing()

    metrics.compute(cars, entry_nodes)
    metrics.save("traffic_data.npz")
    close_window()


if __name__ == "__main__":
    main()
