import numpy as np
from pyray import *

HANDLE_RADIUS = 10

class QuadraticBezier:
    def __init__(self, p0, p1, p2):
        self.points = [
            np.array(p0, dtype=float),
            np.array(p1, dtype=float),
            np.array(p2, dtype=float),
        ]
        self.dragging = None

    def update(self, mouse, pressed, released):
        if pressed and self.dragging is None:
            for i, p in enumerate(self.points):
                if np.linalg.norm(p - mouse) < HANDLE_RADIUS:
                    self.dragging = i
                    break
        if released:
            self.dragging = None
        if self.dragging is not None:
            self.points[self.dragging][:] = mouse

    def draw(self):
        p0, p1, p2 = self.points
        # control lines
        draw_line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]), LIGHTGRAY)
        draw_line(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), LIGHTGRAY)
        # curve
        A = np.array([
            [p0,        p1 - p0        ],
            [p1 - p0,   p0 - 2*p1 + p2],
        ])
        prev = None
        for i in range(101):
            u = np.array([1.0, i / 100.0])
            l = np.einsum('i,ijk,j->k', u, A, u)
            if prev is not None:
                draw_line(int(prev[0]), int(prev[1]), int(l[0]), int(l[1]), BLUE)
            prev = l
        # handles
        colors = [RED, GREEN, RED]
        for p, c in zip(self.points, colors):
            draw_circle_lines(int(p[0]), int(p[1]), HANDLE_RADIUS, c)


# --- global storage ---
objects = [
    QuadraticBezier((100, 100), (300, 500), (500, 100)),
    QuadraticBezier((50, 300), (300, 50), (550, 300)),
]

init_window(600, 600, "Bezier")
set_target_fps(60)

while not window_should_close():
    mp = get_mouse_position()
    mouse   = np.array((mp.x, mp.y), dtype=float)
    pressed  = is_mouse_button_pressed(MOUSE_BUTTON_LEFT)
    released = is_mouse_button_released(MOUSE_BUTTON_LEFT)

    begin_drawing()
    clear_background(WHITE)

    for obj in objects:
        obj.update(mouse, pressed, released)
        obj.draw()

    end_drawing()

close_window()