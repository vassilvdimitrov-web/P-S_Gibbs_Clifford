import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =============================================================================
# PHYSICS
# =============================================================================

# Constants (use k=1 unless specified)
#k = 1.0
G = 1.0


def reduced_mass(m1, m2):
    #This comes directly from the sheet: 1/mu = 1/m1 + 1/m2
    return (m1 * m2) / (m1 + m2)

def angular_momentum(mu, r, v):
    #From the sheet: L = mu * (r x v)
    #In 2D, the cross product is the scalar z-component:
    #L = mu * (r_x * v_y - r_y * v_x)
    return mu * (r[0] * v[1] - r[1] * v[0])
    #return mu * np.cross(r, v)

def energy(mu, r, v, k):
    #total energy of the relative motion:
    #E = (1/2) * mu * |v|^2 - k/|r|
    #where V(r) = -k/r
    kinetic = 0.5 * mu * np.dot(v, v)
    potential = -k / np.linalg.norm(r)
    return kinetic + potential

def effective_potential(r, L, mu, k):
    return L**2 / (2 * mu * r**2) - k / r

def eccentricity_vector(r, v, mu, k):
    #eccentricity vector (equivalent to the
    #Laplace-Runge-Lenz vector divided by mu*k)
    #e = ((mu*|v|^2 - k/|r|)*r - mu*(r·v)*v) / k
    r_norm = np.linalg.norm(r)
    v_sq = np.dot(v, v)
    rv = np.dot(r, v)

    return ((mu * v_sq - k / r_norm) * r - mu * rv * v) / k
    #L_vec = angular_momentum(mu, r, v)
    #return (np.cross(v, L_vec) / k) - (r / np.linalg.norm(r))

def orbit_equation(phi, L, mu, e, k):
    #r(phi) = [L^2/(mu*k)] / [1 + e*cos(phi)]
    #e is the scalar eccentricity = |eccentricity_vector|
    return (L**2 / (mu * k)) / (1 + e * np.cos(phi))

def initial_conditions_from_E_L(E, L, mu, k, phi0=0.0):
    """
    Constructs initial conditions (r0, v0) for the relative motion
    from given total energy E and angular momentum L.

    Let the initial point to be a turning point of the radial motion: r_dot = 0
    and the velocity is purely tangential
    E = L^2/(2*mu*r^2) - k/r
    --> 2E*r^2 + 2k*r - L^2/mu = 0 (quadratic equation for r)
    r0 = r * [cos(phi0), sin(phi0)]

    L = mu * r * v_t  ==> v_t = L / (mu*r)

    The tangential unit vector in polar coordinates: e_phi = [-sin(phi0), cos(phi0)]
    ==> v0 = v_t * e_phi
    """

    if abs(E) < 1e-10:
        # case E = 0
        #2k*r - L^2/mu = 0
        r_turn = L**2 / (2 * mu * k)
    else:
        #2E*r^2 + 2k*r - L^2/mu = 0
        a = 2 * E
        b = 2 * k
        c = -L**2 / mu
        discr = b**2 - 4 * a * c
        if discr < 0:
            raise ValueError("No real orbit exists")
        r1 = (-b + np.sqrt(discr)) / (2 * a)
        r2 = (-b - np.sqrt(discr)) / (2 * a)

        positive_roots = [r for r in (r1, r2) if r > 0]
        if not positive_roots:
            raise ValueError("No positive turning point found.")

        # Choose the smaller positive root
        r_turn = min(positive_roots)

    #the initial position vector at angle phi0
    r0 = r_turn * np.array([np.cos(phi0), np.sin(phi0)])
    #tangential unit vector in polar coordinates
    e_phi = np.array([-np.sin(phi0), np.cos(phi0)])
    #Tangential speed obtained from the angular momentum formula
    v_t = L / (mu * r_turn)
    # Initial velocity is purely tangential
    v0 = v_t * e_phi

    return r0, v0

def vis_viva(r, a, mu_total):
    return np.sqrt(mu_total * (2/r - 1/a))

def semi_major_axis_from_energy(E, mu, k):
    """
    Kepler problem: E = -k/(2a) for bound elliptical orbits in the
    standard relative formulation.
    Relative energy formula is: E = (1/2)mu v^2 - k/r
    --> for an ellipse: a = -k / (2E)
    Only makes sense for E < 0.
    """
    if E >= 0:
        raise ValueError("Semi-major axis is only defined for bound elliptical orbits (E < 0).")
    return -k / (2 * E)

def circular_speed(radius, mu_total):

    #Circular orbital speed at radius r: v_circ = sqrt(mu_total / r)

    #mu_total = G*M where M is the dominant central mass in
    #the m1 >> m2 approximation.

    return np.sqrt(mu_total / radius)

def hohmann_delta_v(r1, r2, mu_total):
    """
    Computes the two delta-v values for a Hohmann transfer between
    circular orbits of radius r1 and r2.
    Transfer ellipse semi-major axis: a_t = (r1 + r2)/2
    Speeds:
        v1 = circular speed on initial orbit
        v2 = circular speed on final orbit
        v_peri = speed at periapsis of transfer ellipse
        v_apo  = speed at apoapsis of transfer ellipse

        delta_v1 = v_peri - v1
        delta_v2 = v2 - v_apo
    """
    a_t = 0.5 * (r1 + r2)

    v1 = np.sqrt(mu_total / r1)
    v2 = np.sqrt(mu_total / r2)

    v_peri = np.sqrt(mu_total * (2 / r1 - 1 / a_t))
    v_apo = np.sqrt(mu_total * (2 / r2 - 1 / a_t))

    delta_v1 = v_peri - v1
    delta_v2 = v2 - v_apo

    return delta_v1, delta_v2, a_t


# =============================================================================
# SIMULATION
# =============================================================================

#k = 1.0

def acceleration(r, mu, k):
    #Relative acceleration for the two-body problem: mu * r_ddot = -k * r / |r|^3
    #r_ddot = -(k/mu) * r / |r|^3
    dist = np.linalg.norm(r)
    return -(k/mu) * r / dist**3

def rk4_step(r, v, dt, mu, k):
    #rk4 step for:
    #r_dot = v
    #v_dot = acceleration(r)
    #returns updated position and velocity after one rk4 step
    def a(pos):
        return acceleration(pos,mu,k)

    k1_v = a(r)
    k1_r = v

    k2_v = a(r + 0.5 * dt * k1_r)
    k2_r = v + 0.5 * dt * k1_v

    k3_v = a(r + 0.5 * dt * k2_r)
    k3_r = v + 0.5 * dt * k2_v

    k4_v = a(r + dt * k3_r)
    k4_r = v + dt * k3_v

    r_new = r + (dt/6)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_new = v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    return r_new, v_new

def simulate_two_body(r0, v0, mu, dt=0.01, steps=5000, k=1.0):
    #Simulate the relative orbit r(t).
    #Returns relative positions and relative velocities
    r = r0.copy().astype(float)
    v = v0.copy().astype(float)
    positions = np.zeros((steps, 2))
    velocities = np.zeros((steps, 2))

    for i in range(steps):
        positions[i] = r
        velocities[i] = v
        r, v = rk4_step(r, v, dt, mu, k)

    return positions, velocities
    """
    positions = []

    for _ in range(steps):
        positions.append(r.copy())
        r, v = rk4_step(r, v, dt)

    return np.array(positions)
    """

def relative_to_com_coordinates(r_traj, m1, m2):
    """
    Converts relative orbit r(t) into the trajectories of the
    two masses in the center-of-mass frame.
    r = x1 - x2, m1*x1 + m2*x2 = 0
    ==> x1 = (m2 / (m1 + m2)) * r
        x2 = -(m1 / (m1 + m2)) * r
    Returns trajectories of mass 1 and mass 2 in the COM frame
    """
    M = m1 + m2
    x1_traj = (m2 / M) * r_traj
    x2_traj = -(m1 / M) * r_traj
    return x1_traj, x2_traj

def simulate_hohmann_transfer(r1, r2, mu_red, mu_total, delta_v1, delta_v2,dt=0.001):
    """
    Simulate a full Hohmann transfer:
    1) first burn at r1
    2) coast for half the transfer ellipse period
    3) second burn at apoapsis
    4) continue on final circular orbit
    """
    k = mu_red * mu_total

    # Initial circular speed
    v_circ1 = np.sqrt(mu_total / r1)

    # Transfer ellipse semi-major axis
    a_transfer = 0.5 * (r1 + r2)

    # Half period of the transfer ellipse
    t_half = np.pi * np.sqrt(a_transfer**3 / mu_total)

    # Number of steps for first phase
    steps1 = int(np.ceil(t_half / dt))

    # Start at periapsis
    r = np.array([r1, 0.0], dtype=float)
    v = np.array([0.0, v_circ1 + delta_v1], dtype=float)

    positions1 = np.zeros((steps1 + 1, 2))
    velocities1 = np.zeros((steps1 + 1, 2))

    positions1[0] = r
    velocities1[0] = v

    for i in range(1, steps1 + 1):
        r, v = rk4_step(r, v, dt, mu_red, k)
        positions1[i] = r
        velocities1[i] = v

    # State after half transfer period = apoapsis
    r_apo = positions1[-1].copy()
    v_apo = velocities1[-1].copy()

    # Second burn in tangential direction
    tangential_dir = v_apo / np.linalg.norm(v_apo)
    v = v_apo + delta_v2 * tangential_dir
    r = r_apo.copy()

    # Simulate one full final circular orbit
    T2 = 2 * np.pi * np.sqrt(r2**3 / mu_total)
    steps2 = int(np.ceil(T2 / dt))

    positions2 = np.zeros((steps2 + 1, 2))
    velocities2 = np.zeros((steps2 + 1, 2))

    positions2[0] = r
    velocities2[0] = v

    for i in range(1, steps2 + 1):
        r, v = rk4_step(r, v, dt, mu_red, k)
        positions2[i] = r
        velocities2[i] = v

    positions_full = np.vstack((positions1, positions2[1:]))
    velocities_full = np.vstack((velocities1, velocities2[1:]))

    return positions_full, velocities_full, positions1
# -----------------------
# THREE BODY (part d)
# -----------------------

def acceleration_3body(pos, masses):
    n = len(masses)
    acc = np.zeros_like(pos)

    for i in range(n):
        for j in range(n):
            if i != j:
                r = pos[j] - pos[i]
                dist = np.linalg.norm(r) + 1e-9
                acc[i] += masses[j] * r / dist**3

    return acc


def simulate_three_body(pos, vel, masses, dt=0.01, steps=5000):
    pos = pos.copy()
    vel = vel.copy()

    traj = np.zeros((steps, len(masses), 2))

    for t in range(steps):
        traj[t] = pos

        # RK4 (important for stability)
        k1_v = acceleration_3body(pos, masses)
        k1_r = vel

        k2_v = acceleration_3body(pos + 0.5*dt*k1_r, masses)
        k2_r = vel + 0.5*dt*k1_v

        k3_v = acceleration_3body(pos + 0.5*dt*k2_r, masses)
        k3_r = vel + 0.5*dt*k2_v

        k4_v = acceleration_3body(pos + dt*k3_r, masses)
        k4_r = vel + dt*k3_v

        pos += (dt/6)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
        vel += (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    return traj


# =============================================================================
# VISUALIZATION
# =============================================================================

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
def animate_three_body(traj, zoom_body=None, zoom_pad=0.2):
    # Main view on the left, optional zoom on `zoom_body` on the right so a
    # very-small orbit (e.g. a heavy mass in Lagrange's equilateral solution)
    # stays visible even when the big picture is dominated by the light bodies.
    if zoom_body is None:
        fig, ax = plt.subplots()
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    span = min(np.max(np.abs(traj)) * 1.1, 5)
    axes[0].set_xlim(-span, span)
    axes[0].set_ylim(-span, span)
    axes[0].set_aspect('equal')
    axes[0].set_title("Full system")

    if zoom_body is not None:
        zb = traj[:, zoom_body, :]
        cx, cy = zb.mean(axis=0)
        r = np.max(np.linalg.norm(zb - [cx, cy], axis=1)) + zoom_pad
        axes[1].set_xlim(cx - r, cx + r)
        axes[1].set_ylim(cy - r, cy + r)
        axes[1].set_aspect('equal')
        axes[1].set_title(f"Zoom on body {zoom_body}")

    lines_main = [axes[0].plot([], [], '-')[0] for _ in range(3)]
    points_main = [axes[0].plot([], [], 'o')[0] for _ in range(3)]

    if zoom_body is not None:
        line_zoom, = axes[1].plot([], [], '-')
        point_zoom, = axes[1].plot([], [], 'o')

    def update(frame):
        for i in range(3):
            x = traj[:frame, i, 0]
            y = traj[:frame, i, 1]
            lines_main[i].set_data(x, y)
            points_main[i].set_data([traj[frame, i, 0]], [traj[frame, i, 1]])

        if zoom_body is not None:
            line_zoom.set_data(traj[:frame, zoom_body, 0], traj[:frame, zoom_body, 1])
            point_zoom.set_data([traj[frame, zoom_body, 0]], [traj[frame, zoom_body, 1]])
            return lines_main + points_main + [line_zoom, point_zoom]
        return lines_main + points_main

    ani = FuncAnimation(fig, update, frames=len(traj), interval=20)
    plt.show()
    return ani

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
        - transfer ellipse (only the transfer half)
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

def animate_hohmann_transfer(full_traj, r1, r2, stride=20, interval=10):
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

    line, = ax.plot([], [], '-', lw=2, color='green')
    point, = ax.plot([], [], 'o', color='red')

    frame_indices = np.arange(0, len(full_traj), stride)

    def update(frame_idx):
        idx = frame_indices[frame_idx]
        x = full_traj[:idx + 1, 0]
        y = full_traj[:idx + 1, 1]
        line.set_data(x, y)
        point.set_data([full_traj[idx, 0]], [full_traj[idx, 1]])
        return line, point

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval)
    plt.show()
    return ani


# =============================================================================
# TESTS / PARTS
# =============================================================================

def part_a():
    # -------------------
    # PART (a): Trajectory
    # -------------------
    # Plot the trajectories of m1 and m2 in the center-of-mass frame for given
    # total energy E and angular momentum L.
    # Plot the Laplace-Runge-Lenz / eccentricity vector.

    m1 = 3
    m2 = 0.9
    mu = reduced_mass(m1, m2)

    k = G * m1 * m2
    #total energy E and angular momentum L
    E = -0.5
    L = 0.875
    #initial conditions from E and L
    r0, v0 = initial_conditions_from_E_L(E, L, mu, k=k, phi0=0.0)
    #simulate relative motion
    traj_rel, vel_rel = simulate_two_body(r0, v0, mu, dt=0.01, steps=5000, k=k)
    # Convert to trajectories of the two masses in the COM frame
    traj1, traj2 = relative_to_com_coordinates(traj_rel, m1, m2)
    #eccentricity vector
    e_vec = eccentricity_vector(r0, v0, mu, k)

    # Diagnostics
    print("=== PART (a) ===")
    print("Initial conditions from E and L")
    print("r0 =", r0)
    print("v0 =", v0)
    print("Energy check =", energy(mu, r0, v0, k))
    print("Angular momentum check =", angular_momentum(mu, r0, v0))
    print("Eccentricity =", np.linalg.norm(e_vec))

    # Plot result for task (a)
    plot_two_body_com(traj1, traj2, e_vec=e_vec, title="Part (a): Two-body trajectories in COM frame")
    """
    r0 = np.array([1.0, 0.0])
    v0 = np.array([0.0, 1.0])

    traj = simulate_two_body(r0, v0)
    plot_trajectory(traj, "Elliptical Orbit")
    """

def part_b():
    # -------------------
    # PART (b): Vis-viva
    # -------------------
    # Plot how the relative velocity depends on the angle phi.

    ## ===== Simulation from Part A
    m1 = 3
    m2 = 0.9
    mu = reduced_mass(m1, m2)

    k = G * m1 * m2
    #total energy E and angular momentum L
    E = -0.5
    L = 0.875
    #initial conditions from E and L
    r0, v0 = initial_conditions_from_E_L(E, L, mu, k=k, phi0=0.0)

    #eccentricity vector
    e_vec = eccentricity_vector(r0, v0, mu, k)
    ## ===== Simulation from Part A

    # Use the same orbit as in part (a)
    e = np.linalg.norm(e_vec)

    # Semi-major axis from the energy
    a = semi_major_axis_from_energy(E, mu, k)

    # mu_total = k / mu = G*(m1+m2)
    mu_total = k / mu

    # Create phi values
    phi_vals = np.linspace(0, 2 * np.pi, 500)

    #compute r(phi) from the orbit equation
    r_vals = orbit_equation(phi_vals, L, mu, e, k)

    # Compute v(phi) from vis-viva
    v_vals = vis_viva(r_vals, a, mu_total)

    # Plot result for task (b)
    plot_vis_viva(phi_vals, v_vals)

    """
    r_vals = np.linspace(0.5, 3, 100)
    a = 1.5
    v_vals = np.sqrt(1 * (2/r_vals - 1/a))

    from visualization import plot_vis_viva
    plot_vis_viva(r_vals, v_vals)
    """

def part_c():
    # We now assume m1 >> m2
    M_central = 10.0
    m_sat = 0.001

    mu_red_transfer = reduced_mass(M_central, m_sat)
    mu_total_transfer = G * (M_central + m_sat)

    # Initial and final circular radii
    r1 = 1.0
    r2 = 4.0

    # Compute Hohmann transfer data
    delta_v1, delta_v2, a_transfer = hohmann_delta_v(r1, r2, mu_total_transfer)

    print("delta_v1 =", delta_v1)
    print("delta_v2 =", delta_v2)
    print("transfer semi-major axis =", a_transfer)

    transfer_traj, transfer_vel, transfer_only = simulate_hohmann_transfer(
    r1, r2,
    mu_red_transfer, mu_total_transfer,
    delta_v1, delta_v2,
    dt=0.001
)

    theta = np.linspace(0, 2 * np.pi, 500)
    circle1 = np.column_stack((r1 * np.cos(theta), r1 * np.sin(theta)))
    circle2 = np.column_stack((r2 * np.cos(theta), r2 * np.sin(theta)))

    plot_hohmann_transfer(circle1, circle2, transfer_only, title="Part (c): Hohmann transfer")
    ani_hohmann = animate_hohmann_transfer(transfer_traj, r1, r2, stride=20, interval=10)

def part_d():
    # -------------------
    # PART (d): Three-body
    # -------------------
    # pos = np.array([[1,0], [-1,0], [0,1]], dtype=float)
    # vel = np.array([[0,0.5], [0,-0.5], [-0.5,0]], dtype=float)
    # masses = np.array([1,1,1])

    # traj3 = simulate_three_body(pos, vel, masses)
    # plot_multiple([traj3[:,0], traj3[:,1], traj3[:,2]])

    # --- Case 1: Equal masses (figure-8)
    # Circular, periodic and stable orbit (very simple & predictable)
    masses = np.array([1, 1, 1])

    pos = np.array([
        [0, 1],
        [0, 0],
        [0, -1]
    ], dtype=float)

    vel = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
        [-1.0, 0.0]
    ])

    masses = np.array([1, 1, 1])

    traj = simulate_three_body(pos, vel, masses, dt=0.01, steps=8000)

    ani_d2 = animate_three_body(traj)

    # Less trivial periodic config, looks stable initially, but is not completely stable,
    # you can see that in later iterations the masses move in a certain direction
    masses = np.array([1, 1, 1])
    pos = np.array([
        [1, 0],
        [-0.5,0.8660],
        [-0.5,-0.8660]
    ], dtype=float)

    vel = np.array([
        [0.0,1],
        [-0.8660, -0.5],
        [0.8660, -0.5]
    ])

    vel = vel / np.sqrt(2)

    traj = simulate_three_body(pos, vel, masses, dt=0.01, steps=10000)

    ani_d3 = animate_three_body(traj)

    # --- Case 2: Lagrange equilateral triangle with a very heavy body
    # Lagrange's solution is periodic for any mass ratio: the triangle rotates
    # rigidly and each body traces a circle about the common center of mass. With M >> m
    # the heavy body's circle is small (radius = 3m/(M+2m) for side sqrt(3)),
    # because momentum conservation forbids a large excursion of the heavy body.
    # We can see a periodic orbit the zoomed sub-plot.
    masses = np.array([100.0, 1.0, 1.0])

    pos_centroid = np.array([
        [1.0, 0.0],
        [-0.5, np.sqrt(3)/2],
        [-0.5, -np.sqrt(3)/2],
    ])

    # Shift so the COM is at the origin
    com = (masses[:, None] * pos_centroid).sum(axis=0) / masses.sum()
    pos = pos_centroid - com

    # Exact circular velocity for the equilateral configuration
    a_side = np.sqrt(3)
    omega = np.sqrt(G * masses.sum() / a_side**3)
    vel = omega * np.column_stack((-pos[:, 1], pos[:, 0]))

    traj = simulate_three_body(pos, vel, masses, dt=0.002, steps=6000)

    # zoom_body=0 zooms on the heavy body so its small but exact circle is seen
    ani_d4 = animate_three_body(traj, zoom_body=0)

    # --- Case 3: Hierarchical triple with m1 >> m2 >> m3 (three distinct orbits)
    # A Lagrange-like configuration forces all three bodies to share one triangle
    # and therefore one common length scale. To get THREE different orbital paths
    # we use a hierarchical / Kepler-of-Keplers setup:
    #   - m1 and m2 form a tight inner binary, circular about their own COM.
    #   - The inner binary's COM and m3 form an outer binary, circular about the
    #     full-system COM.
    # a_out is chosen so that the outer period is an exact integer multiple of the
    # inner period (here 8:1 mean-motion resonance), making the full three-body
    # motion periodic in the COM frame. Each body traces its own distinct path:
    # m1 a tiny wobble, m2 a medium circle, m3 a large outer circle.
    masses = np.array([1.0, 0.1, 0.001])   # m1 >> m2 >> m3
    m1, m2, m3 = masses
    M_in = m1 + m2
    M_tot = M_in + m3

    a_in = 1.0                              # inner binary separation
    # (omega_in / omega_out)^2 = (M_in/M_tot) * (a_out/a_in)^3 -> pick ratio = 8
    a_out = a_in * (64.0 * M_tot / M_in) ** (1.0/3.0)

    omega_in = np.sqrt(G * M_in / a_in**3)
    omega_out = np.sqrt(G * M_tot / a_out**3)

    # Inner-binary geometry (distances from inner-binary COM)
    r1_in = a_in * m2 / M_in   # m1's swing radius
    r2_in = a_in * m1 / M_in   # m2's swing radius

    # Outer-binary geometry (distances from full-system COM)
    d_in = a_out * m3 / M_tot      # inner-COM offset from full COM
    d_out = a_out * M_in / M_tot   # m3's distance from full COM (opposite side)

    # Inner binary on the -x side, m3 on the +x side
    inner_com = np.array([-d_in, 0.0])
    pos = np.array([
        inner_com + np.array([-r1_in, 0.0]),  # m1
        inner_com + np.array([ r2_in, 0.0]),  # m2
        np.array([ d_out, 0.0]),              # m3
    ])

    # Velocities: outer orbit is CCW -> m3 moves +y, inner-COM moves -y.
    # The inner binary also rotates CCW about its own COM.
    v_inner_com_y = -omega_out * d_in
    v_m3_y        =  omega_out * d_out
    vel = np.array([
        [0.0, v_inner_com_y - omega_in * r1_in],  # m1
        [0.0, v_inner_com_y + omega_in * r2_in],  # m2
        [0.0, v_m3_y],                            # m3
    ])

    traj = simulate_three_body(pos, vel, masses, dt=0.01, steps=15000)

    # Zoom on m1 so its tiny wobble is visible alongside the full system view.
    ani_d5 = animate_three_body(traj, zoom_body=0)


PARTS = {
    "a": part_a,
    "b": part_b,
    "c": part_c,
    "d": part_d,
}


def main():
    parser = argparse.ArgumentParser(description="Run a specific project part.")
    parser.add_argument(
        "part",
        choices=sorted(PARTS.keys()),
        help="Which part to run (a, b, c, or d).",
    )
    args = parser.parse_args()
    PARTS[args.part]()


if __name__ == "__main__":
    main()
