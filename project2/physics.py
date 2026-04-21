import numpy as np

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