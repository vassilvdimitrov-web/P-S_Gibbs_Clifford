import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# -------------------
# Single trajectory
# -------------------
def plot_trajectory(positions, title="Orbit"):
    plt.figure()
    plt.plot(positions[:, 0], positions[:, 1], label="trajectory")
    plt.scatter([0], [0],color='black', label="Central body")
    plt.axis("equal")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


# -------------------
# Multiple trajectories
# -------------------
"""
def plot_multiple(trajectories):
    plt.figure()
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1])
    plt.axis("equal")
    plt.title("Three-body trajectories")
    plt.show()
"""
def plot_multiple(trajectories, labels=None, title="Trajectories"):
    """
    Used for:
        (a) plotting both masses in the COM frame
        (c) plotting initial orbit, final orbit, transfer orbit
        (d) plotting the three bodies
    """
    plt.figure()

    for i, traj in enumerate(trajectories):
        if labels is None:
            plt.plot(traj[:, 0], traj[:, 1])
        else:
            plt.plot(traj[:, 0], traj[:, 1], label=labels[i])

    plt.axis("equal")
    plt.title(title)
    if labels is not None:
        plt.legend(loc="upper right")
    plt.grid(True)
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

# ------------------------------------------------------------
# Special plot for task (a)
# ------------------------------------------------------------
def plot_two_body_com(x1_traj, x2_traj, e_vec=None, title="Two-body orbit in COM frame"):
    """
    Plot the trajectories of both masses in the center-of-mass frame.
    Also draws the eccentricity / LRL vector if provided.
    """
    plt.figure(figsize=(7, 7))

    plt.plot(x1_traj[:, 0], x1_traj[:, 1], label="mass m1")
    plt.plot(x2_traj[:, 0], x2_traj[:, 1], label="mass m2")

    # Initial positions
    plt.scatter(x1_traj[0, 0], x1_traj[0, 1], s=40)
    plt.scatter(x2_traj[0, 0], x2_traj[0, 1], s=40)

    # Center of mass
    plt.scatter([0], [0], color='black', label="center of mass")

    # Eccentricity vector / LRL vector direction
    if e_vec is not None:
        plt.arrow(
            0, 0,
            e_vec[0], e_vec[1],
            head_width=0.05,
            length_includes_head=True,
            color='red'
        )
        plt.plot([], [], color='red', label="eccentricity / LRL direction")

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

# -------------------
# Vis-viva plot
# -------------------
def plot_vis_viva(phi_vals, v_vals):
    #Plot the relative speed v as a function of the angle phi.
    plt.figure()
    plt.plot(phi_vals, v_vals)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$v(\phi)$")
    plt.title("Vis-viva relation")
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------
# Special plot for task (c)
# ------------------------------------------------------------
def plot_hohmann_transfer(circle1, circle2, transfer, title="Hohmann transfer"):
    """
    Plot:
        - initial circular orbit
        - final circular orbit
        - transfer ellipse
    """
    plt.figure(figsize=(7, 7))

    plt.plot(circle1[:, 0], circle1[:, 1], label="initial circular orbit")
    plt.plot(circle2[:, 0], circle2[:, 1], label="final circular orbit")
    plt.plot(transfer[:, 0], transfer[:, 1], label="transfer ellipse")

    plt.scatter([0], [0], color='black', label="central body")
    plt.axis("equal")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

def animate_hohmann_transfer(transfer_traj, r1, r2):
    #Animate the motion along the transfer ellipse

    fig, ax = plt.subplots(figsize=(7, 7))

    theta = np.linspace(0, 2 * np.pi, 400)
    circle1 = np.column_stack((r1 * np.cos(theta), r1 * np.sin(theta)))
    circle2 = np.column_stack((r2 * np.cos(theta), r2 * np.sin(theta)))

    ax.plot(circle1[:, 0], circle1[:, 1], '--', label="initial circular orbit")
    ax.plot(circle2[:, 0], circle2[:, 1], '--', label="final circular orbit")
    ax.scatter([0], [0], color='black', label="central body")

    ax.set_xlim(-1.2 * r2, 1.2 * r2)
    ax.set_ylim(-1.2 * r2, 1.2 * r2)
    ax.set_aspect('equal')
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("Hohmann transfer animation")

    line, = ax.plot([], [], '-', lw=2)
    point, = ax.plot([], [], 'o')

    def update(frame):
        x = transfer_traj[:frame, 0]
        y = transfer_traj[:frame, 1]

        line.set_data(x, y)
        point.set_data([transfer_traj[frame, 0]], [transfer_traj[frame, 1]])
        return line, point

    ani = FuncAnimation(fig, update, frames=len(transfer_traj), interval=20)
    plt.show()
