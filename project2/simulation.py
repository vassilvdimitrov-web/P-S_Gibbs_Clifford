import numpy as np

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

def simulate_hohmann_transfer(r1, r2, mu_red, mu_total, delta_v1, dt=0.01, steps=4000):
    """
    Simulate the transfer ellipse after the first burn.
    We assume m1 >> m2, so the large body is fixed and the smaller 
    body moves in the central field.

    Start on a circular orbit of radius r1 at:
        r0 = [r1, 0]
        v0 = [0, v_circ + delta_v1]
    Then integrate the motion numerically.

    r1: Radius of initial circular orbit
    r2: Radius of target circular orbit
    mu_red: Reduced mass of the two-body system
    mu_total: Gravitational parameter G*M of the dominant central body
    delta_v1: First Hohmann burn
    """
    #acceleration is -(k/mu_red) r/|r|^3
    #we want this to equal -mu_total * r / |r|^3
    # --> k = mu_red * mu_total
    k = mu_red * mu_total

    v_circ1 = np.sqrt(mu_total / r1)

    r0 = np.array([r1, 0.0])
    v0 = np.array([0.0, v_circ1 + delta_v1])

    return simulate_two_body(r0, v0, mu_red, dt=dt, steps=steps, k=k)
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