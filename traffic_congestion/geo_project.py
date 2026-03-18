import math
import numpy as np
from pyray import *
from enum import Enum, auto

NODE_RADIUS   = 14
ARROW_SIZE    = 14
HANDLE_RADIUS = 8

class QuadraticBezier:
    def __init__(self, p0, p2):
        self.p0   = np.array(p0, dtype=float)
        self.p2   = np.array(p2, dtype=float)
        self.ctrl = (self.p0 + self.p2) / 2
        self._dragging_ctrl = False

    def update_endpoints(self, p0, p2):
        self.p0[:] = p0
        self.p2[:] = p2

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

    def point_at(self, t):
        return (1 - t)**2 * self.p0 + 2 * (1 - t) * t * self.ctrl + t**2 * self.p2

    def tangent_at(self, t):
        return 2 * (1 - t) * (self.ctrl - self.p0) + 2 * t * (self.p2 - self.ctrl)

    def draw(self, color=BLACK):
        steps = 100
        prev  = None
        tip   = None
        tip_t = None

        for i in range(steps + 1):
            t  = i / steps
            pt = self.point_at(t)

            if np.linalg.norm(pt - self.p0) < NODE_RADIUS:
                prev = pt
                continue
            if np.linalg.norm(pt - self.p2) < NODE_RADIUS:
                tip, tip_t = pt, t
                break

            if prev is not None:
                draw_line(int(prev[0]), int(prev[1]), int(pt[0]), int(pt[1]), color)
            prev = pt

        if tip is None:
            return

        tang   = self.tangent_at(tip_t)
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
                    if not any(e.p0 is nodes[drag_start] and e.p2 is nodes[node_hit]
                               for e in edges):
                        edges.append(QuadraticBezier(nodes[drag_start], nodes[node_hit]))
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
        preview = QuadraticBezier(nodes[drag_start], mp)
        preview.draw(GRAY)

    for edge in edges:
        edge.draw()
        edge.draw_handle()

    for i, node in enumerate(nodes):
        color = DARKBLUE if i == drag_start else BLUE
        draw_circle(int(node[0]), int(node[1]), NODE_RADIUS, color)

    draw_circle(int(mp[0]), int(mp[1]), 6, RED)

    end_drawing()

close_window()