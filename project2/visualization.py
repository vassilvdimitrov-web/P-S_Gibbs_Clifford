import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# -------------------
# Single trajectory
# -------------------
def plot_trajectory(positions, title="Orbit"):
    plt.figure()
    plt.plot(positions[:, 0], positions[:, 1])
    plt.scatter([0], [0], label="Central body")
    plt.axis("equal")
    plt.title(title)
    plt.legend()
    plt.show()


# -------------------
# Multiple trajectories
# -------------------
def plot_multiple(trajectories):
    plt.figure()
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1])
    plt.axis("equal")
    plt.title("Three-body trajectories")
    plt.show()


# -------------------
# Animation (PART D)
# -------------------
def animate_three_body(traj):
    fig, ax = plt.subplots()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    lines = [ax.plot([], [], '-')[0] for _ in range(3)]
    points = [ax.plot([], [], 'o')[0] for _ in range(3)]

    def update(frame):
        for i in range(3):
            x = traj[:frame, i, 0]
            y = traj[:frame, i, 1]

            lines[i].set_data(x, y)
            points[i].set_data(traj[frame, i, 0], traj[frame, i, 1])

        return lines + points

    ani = FuncAnimation(fig, update, frames=len(traj), interval=20)
    plt.show()


# -------------------
# Vis-viva plot
# -------------------
def plot_vis_viva(r_vals, v_vals):
    plt.figure()
    plt.plot(r_vals, v_vals)
    plt.xlabel("r")
    plt.ylabel("v")
    plt.title("Vis-viva relation")
    plt.show()