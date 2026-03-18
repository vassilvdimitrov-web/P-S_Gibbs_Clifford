import math
import numpy as np
from pyray import *
from enum import Enum, auto
import pickle

NODE_RADIUS   = 14
ARROW_SIZE    = 14
HANDLE_RADIUS = 8

class QuadraticBezier:
    def __init__(self, node0, node1, nodes):
        self.node0 = node0
        self.node1 = node1
        self.ctrl = (nodes[node0] + nodes[node1]) / 2
        self._dragging_ctrl = False

    def try_grab(self, mouse):
        if np.linalg.norm(self.ctrl - mouse) < HANDLE_RADIUS:
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
        return (1 - t)**2 * nodes[self.node0] + 2 * (1 - t) * t * self.ctrl + t**2 * nodes[self.node1]

    def tangent_at(self, t, nodes):
        return 2 * (1 - t) * (self.ctrl - nodes[self.node0]) + 2 * t * (nodes[self.node1] - self.ctrl)

    def draw(self, nodes, color=BLACK):
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
        draw_circle(int(self.ctrl[0]), int(self.ctrl[1]), HANDLE_RADIUS, GREEN)


class State(Enum):
    IDLE     = auto()
    DRAGGING = auto()


def find_node_at(nodes, pos, radius=NODE_RADIUS):
    for i, node in enumerate(nodes):
        if check_collision_point_circle(pos, Vector2(node[0], node[1]), radius):
            return i
    return None


def update(state, drag_start, nodes, edges, mouse_pos, node_hit):
    mp       = np.array([mouse_pos.x, mouse_pos.y])
    pressed  = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    released = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)

    # let edges handle control-point dragging first
    for edge in edges:
        if pressed:
            edge.try_grab(mp)
        edge.drag(mp)
        if released:
            edge.release()

    match state:
        case State.IDLE:
            if pressed:
                ctrl_grabbed = any(e.is_dragging for e in edges)
                if not ctrl_grabbed:
                    if node_hit is not None:
                        return State.DRAGGING, node_hit
                    else:
                        nodes.append(mp.copy())

        case State.DRAGGING:
            if released:
                if node_hit is not None and node_hit != drag_start:
                    # FIX 1: use node0/node1 (not e.p0/e.p2) for duplicate-edge check
                    if not any(e.node0 == drag_start and e.node1 == node_hit
                               for e in edges):
                        edges.append(QuadraticBezier(drag_start, node_hit, nodes))
                return State.IDLE, None

    return state, drag_start


init_window(600, 600, "Directed Graph")
set_target_fps(60)

nodes      = []
edges      = []
state      = State.IDLE
drag_start = None

while not window_should_close():
    begin_drawing()
    clear_background(WHITE)

    mouse_pos = get_mouse_position()
    node_hit  = find_node_at(nodes, mouse_pos)
    mp        = np.array([mouse_pos.x, mouse_pos.y])

    state, drag_start = update(state, drag_start, nodes, edges, mouse_pos, node_hit)

    if state == State.DRAGGING:
        # FIX 2: removed stray print(); FIX 3: pass nodes first, color second
        preview = QuadraticBezier(drag_start, -1, nodes + [mp])
        preview.draw(nodes + [mp], GRAY)

    for edge in edges:
        edge.draw(nodes)
        edge.draw_handle()

    for i, node in enumerate(nodes):
        color = DARKBLUE if i == drag_start else BLUE
        draw_circle(int(node[0]), int(node[1]), NODE_RADIUS, color)

    if is_key_pressed(KeyboardKey.KEY_S):
        data = {
            "nodes": nodes,
            "edges": edges,
        }
        pickle.dump(data, open('data.pkl', 'wb'))

    if is_key_pressed(KeyboardKey.KEY_L):
        data = pickle.load(open('data.pkl', 'rb'))
        nodes = data["nodes"]
        edges = data["edges"]

    draw_circle(int(mp[0]), int(mp[1]), 6, RED)

    end_drawing()

close_window()