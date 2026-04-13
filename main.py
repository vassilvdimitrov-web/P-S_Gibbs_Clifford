import numpy as np
from simulation import simulate_two_body, simulate_three_body
from visualization import plot_trajectory, plot_multiple
from physics import eccentricity_vector

# -------------------
# PART (a): Trajectory
# -------------------
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])

traj = simulate_two_body(r0, v0)
plot_trajectory(traj, "Elliptical Orbit")

# -------------------
# PART (b): Vis-viva
# -------------------
r_vals = np.linspace(0.5, 3, 100)
a = 1.5
v_vals = np.sqrt(1 * (2/r_vals - 1/a))

from visualization import plot_vis_viva
plot_vis_viva(r_vals, v_vals)

# -------------------
# PART (c): Hohmann (simplified visualization)
# -------------------
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])

traj1 = simulate_two_body(r0, v0, steps=2000)

r0_new = np.array([2.0, 0.0])
v0_new = np.array([0.0, 0.7])

traj2 = simulate_two_body(r0_new, v0_new, steps=2000)

plot_multiple([traj1, traj2])

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