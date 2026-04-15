import numpy as np

k = 1.0

def acceleration(r):
    dist = np.linalg.norm(r)
    return -k * r / dist**3

def rk4_step(r, v, dt):
    def a(pos):
        return acceleration(pos)

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

def simulate_two_body(r0, v0, dt=0.01, steps=5000):
    r = r0.copy()
    v = v0.copy()

    positions = []

    for _ in range(steps):
        positions.append(r.copy())
        r, v = rk4_step(r, v, dt)

    return np.array(positions)

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

def total_energy(pos, vel, masses):
    KE = 0
    PE = 0
    n = len(masses)

    for i in range(n):
        KE += 0.5 * masses[i] * np.dot(vel[i], vel[i])
        for j in range(i+1, n):
            r = np.linalg.norm(pos[i] - pos[j]) + 1e-9 #avoid 0 division
            PE -= masses[i]*masses[j]/r

    return KE + PE