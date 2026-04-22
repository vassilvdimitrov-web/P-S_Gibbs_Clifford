import numpy as np
from simulation import simulate_two_body, simulate_three_body, relative_to_com_coordinates, simulate_hohmann_transfer
from visualization import plot_trajectory, plot_multiple, plot_two_body_com, plot_vis_viva, plot_hohmann_transfer,animate_hohmann_transfer
from physics import G,reduced_mass,angular_momentum,energy,eccentricity_vector, initial_conditions_from_E_L,orbit_equation,vis_viva,semi_major_axis_from_energy,hohmann_delta_v


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
    from visualization import animate_three_body

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

