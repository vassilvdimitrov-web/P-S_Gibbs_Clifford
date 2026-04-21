import numpy as np
#from simulation import simulate_two_body, simulate_three_body
#from visualization import plot_trajectory, plot_multiple
#from physics import eccentricity_vector
from simulation import simulate_two_body, simulate_three_body, relative_to_com_coordinates, simulate_hohmann_transfer
from visualization import plot_trajectory, plot_multiple, plot_two_body_com, plot_vis_viva, plot_hohmann_transfer,animate_hohmann_transfer
from physics import G,reduced_mass,angular_momentum,energy,eccentricity_vector, initial_conditions_from_E_L,orbit_equation,vis_viva,semi_major_axis_from_energy,hohmann_delta_v

# -------------------
# PART (a): Trajectory
# -------------------
# Plot the trajectories of m1 and m2 in the center-of-mass frame for given 
# total energy E and angular momentum L.
# Plot the Laplace-Runge-Lenz / eccentricity vector.

m1 = 2.0
m2 = 1.0
mu = reduced_mass(m1, m2)

k = G * m1 * m2
#total energy E and angular momentum L
E = -0.4
L = 0.9
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
# -------------------
# PART (b): Vis-viva
# -------------------
# Plot how the relative velocity depends on the angle phi.

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
# -------------------
# PART (c): Hohmann (simplified visualization)
# -------------------
# Transfer from one circular orbit of radius r1 to another circular 
# orbit of radius r2 around a dominant central mass m1.

# We now assume m1 >> m2
M_central = 10.0
m_sat = 0.001

mu_red_transfer = reduced_mass(M_central, m_sat)
mu_total_transfer = G * (M_central + m_sat)

# Initial and final circular radii
r1 = 1.0
r2 = 2.0

# Compute Hohmann transfer data
delta_v1, delta_v2, a_transfer = hohmann_delta_v(r1, r2, mu_total_transfer)

print("delta_v1 =", delta_v1)
print("delta_v2 =", delta_v2)
print("transfer semi-major axis =", a_transfer)

# Simulate the transfer ellipse after the first burn
transfer_traj, transfer_vel = simulate_hohmann_transfer(
    r1, r2, mu_red_transfer, mu_total_transfer, delta_v1,
    dt=0.01, steps=2500
)

# Build the two circular reference orbits for plotting
theta = np.linspace(0, 2 * np.pi, 500)
circle1 = np.column_stack((r1 * np.cos(theta), r1 * np.sin(theta)))
circle2 = np.column_stack((r2 * np.cos(theta), r2 * np.sin(theta)))

# Plot result for task (c)
plot_hohmann_transfer(circle1, circle2, transfer_traj, title="Part (c): Hohmann transfer")
# Animate result for task (c)
animate_hohmann_transfer(transfer_traj, r1, r2)

"""
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])

traj1 = simulate_two_body(r0, v0, steps=2000)

r0_new = np.array([2.0, 0.0])
v0_new = np.array([0.0, 0.7])

traj2 = simulate_two_body(r0_new, v0_new, steps=2000)

plot_multiple([traj1, traj2])
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