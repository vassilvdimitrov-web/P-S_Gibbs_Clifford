import numpy as np
from simulation import simulate_two_body, simulate_three_body
from visualization import plot_trajectory, plot_multiple
from physics import eccentricity_vector, reconstruct_positions
from visualization import plot_vis_viva_phi, plot_vis_viva
# -------------------
# PART (a): Trajectory
# -------------------
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])

traj = simulate_two_body(r0, v0)
plot_trajectory(traj, "Elliptical Orbit")

m1, m2 = 2.0, 1.0

r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.2])

traj_r = simulate_two_body(r0, v0)

x1, x2 = reconstruct_positions(traj_r, m1, m2)

plot_multiple([x1, x2])

# -------------------
# PART (b): Vis-viva
# -------------------
import matplotlib.pyplot as plt

phi = np.linspace(0, 2*np.pi, 200)

L = 1.0
mu = 1.0
e = 0.5
a = 1.5

r_vals = (L**2 / (mu * 1.0)) / (1 + e * np.cos(phi))
v_vals = np.sqrt(1.0 * (2/r_vals - 1/a))

plot_vis_viva_phi(phi, v_vals)

# -------------------
# PART (c): Hohmann (simplified visualization)
# -------------------
r1 = 1.0
r2 = 2.0

# initial positions
r0 = np.array([r1, 0.0])

# circular velocities
v1 = np.sqrt(1/r1)
v2 = np.sqrt(1/r2)

# transfer orbit semi-major axis
a = (r1 + r2) / 2

# velocities at periapsis & apoapsis
v_peri = np.sqrt(1 * (2/r1 - 1/a))
v_apo = np.sqrt(1 * (2/r2 - 1/a))

# --- 1. Initial circular orbit ---
traj_circ1 = simulate_two_body(r0, np.array([0.0, v1]), steps=2000)

# --- 2. Transfer orbit ---
traj_transfer = simulate_two_body(r0, np.array([0.0, v_peri]), steps=2000)

# --- 3. Final circular orbit (start at r2) ---
r0_final = np.array([r2, 0.0])
traj_circ2 = simulate_two_body(r0_final, np.array([0.0, v2]), steps=2000)

# plot all together
plot_multiple([traj_circ1, traj_transfer, traj_circ2])


""" 
# -------------------
# PART (d): Three-body
# -------------------
pos = np.array([[1,0], [-1,0], [0,1]], dtype=float)
vel = np.array([[0,0.5], [0,-0.5], [-0.5,0]], dtype=float)
masses = np.array([1,1,1])

traj3 = simulate_three_body(pos, vel, masses)
plot_multiple([traj3[:,0], traj3[:,1], traj3[:,2]])
from visualization import animate_three_body

# Choose ONE case at a time

# --- Case 1: Equal masses (figure-8)
masses = np.array([1, 1, 1])

pos = np.array([
    [-1, 3],
    [2, 0],
    [3, 2]
], dtype=float)

vel = np.array([
    [0.347111, 0.532728],
    [0.347111, 0.532728],
    [-0.694222, -1.065456]
])

traj = simulate_three_body(pos, vel, masses, dt=0.1, steps=8000)

animate_three_body(traj)

"""

# -------------------
# PART (d): Three-body
# -------------------

from simulation import total_energy

cases = [
    ("Equal masses", np.array([1, 1, 1])),
    ("One dominant", np.array([100, 1, 1])),
    ("Hierarchical", np.array([100, 10, 1]))
]

for title, masses in cases:

    print(f"\nRunning case: {title}")

    pos = np.array([[1,0], [-1,0], [0,1]], dtype=float)
    vel = np.array([[0,0.5], [0,-0.5], [-0.5,0]], dtype=float)

    traj = simulate_three_body(pos, vel, masses, dt=0.01, steps=3000)

    # Plot trajectories
    plot_multiple([traj[:,0], traj[:,1], traj[:,2]])

    # Energy check (initial vs final)
    E_initial = total_energy(pos, vel, masses)
    E_final = total_energy(traj[-1], vel, masses)

    print(f"Initial Energy: {E_initial:.5f}")
    print(f"Final Energy:   {E_final:.5f}")