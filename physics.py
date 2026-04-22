import numpy as np

# Constants (use k=1 unless specified)
k = 1.0

def reduced_mass(m1, m2):
    return (m1 * m2) / (m1 + m2)

def angular_momentum(mu, r, v):
    return mu * np.cross(r, v)

def energy(mu, r, v):
    kinetic = 0.5 * mu * np.dot(v, v)
    potential = -k / np.linalg.norm(r)
    return kinetic + potential

def effective_potential(r, L, mu):
    return L**2 / (2 * mu * r**2) - k / r

def eccentricity_vector(r, v, mu):
    # convert to 3D
    r3 = np.array([r[0], r[1], 0.0])
    v3 = np.array([v[0], v[1], 0.0])

    L_vec = np.cross(mu * r3, v3)  # angular momentum vector

    e_vec_3d = (np.cross(v3, L_vec) / (k)) - (r3 / np.linalg.norm(r3))

    return e_vec_3d[:2]  # return back to 2D
def orbit_equation(phi, L, mu, e):
    return (L**2 / (mu * k)) / (1 + e * np.cos(phi))

def vis_viva(r, a):
    return np.sqrt(k * (2/r - 1/a))

def reconstruct_positions(r_traj, m1, m2):
    M = m1 + m2
    x1 = (m2 / M) * r_traj
    x2 = -(m1 / M) * r_traj
    return x1, x2