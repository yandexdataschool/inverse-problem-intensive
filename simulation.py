import jax.numpy as np
from jax.ops import index_add
from jax.api import jit
from jax_md import space
from jax_md import simulate, quantity
from utils import lennard_jones_pair_no_cutoff

boltzman_k = 1.380649e-23 # J/K
eV = 1.602176634e-19 # eV/J
# Units
sigma_0 = 1e-10 # m
mu_0 = 58e-27 # kg
tau_0 = 4e-14 # s
E_0 = mu_0*(sigma_0/tau_0)**2 # J
K_B_adjusted = boltzman_k / E_0
N_side = 5
N_A = N_side**2
N_B = (N_side-1)**2

species = np.array([0]*N_A + [1]*N_B, dtype=np.int32)

default_sigma = np.array([
        [0.9, 0.6],
        [0.6, 0.865]
    ], dtype=np.float32)

default_epsilon = np.array([
        [1.14, 0.217],
        [0.217, 1.]], dtype=np.float32)


displacement_fun, shift_fun = space.free()

dt = np.float32(1e-2)
def simulate_with_parameters(
    epsilon=default_epsilon,
    sigma=default_sigma):
    # To reproduce the paper in all its surface enrgy glory,
    # we compute all the pairwise interactions without a cutoff
    energy_fun = lennard_jones_pair_no_cutoff(
        displacement_fun, species=species, sigma=sigma, epsilon=epsilon)
    init_fun, apply_fun = simulate.nve(energy_fun, shift_fun, dt=dt)
    return init_fun, apply_fun, energy_fun

def get_temperature(state):
    """Returns temperature in K."""
    return quantity.temperature(state.velocity, state.mass)/K_B_adjusted


def compute_angle_cos(a, b, c):
    A2 = np.sum((b - c)**2, axis=1)
    C2 = np.sum((a - b)**2, axis=1)
    B2 = np.sum((a - c)**2, axis=1)
    return (B2+C2-A2)*0.5/np.sqrt(C2*B2)


def compute_phase_angle_cos(posistions, N_A):
    # N - 1     N^2 - N 
    # ...
    # 0         N^2 - 1
    # Will horribly fail for the differently indexed lattices
    N = np.int32(np.sqrt(N_A))
    a_index = np.arange(N_A).reshape((N, N))[:-1, :-1].ravel()
    b_index = np.arange(N_A).reshape((N, N))[1:, :-1].ravel()
    c_index = np.arange(N_A).reshape((N, N))[:-1, 1:].ravel()
    return compute_angle_cos(posistions[a_index], posistions[b_index], posistions[c_index])


def compute_normalised_phase(positions, N_A, norms):
    all_cos = compute_phase_angle_cos(positions, N_A)
    # Sometimes we catch more than one cell
    expanded_norms = np.expand_dims(norms, 1)
    all_phases = np.arccos(all_cos) % expanded_norms
    adjusted_phases = np.minimum(all_phases, expanded_norms-all_phases)
    return adjusted_phases.mean(axis=1), adjusted_phases.std(axis=1)

                                             
def get_angular_momentum(position, mass, velocity):
    return np.cross(position, (mass * velocity)).sum()

def get_angular_momentum_state(state):
    return get_angular_momentum(state.position, state.mass, state.velocity)

def get_perpendicular(a):
    # a is assumed to be a 3D array with two last dimensions being the vector
    # TODO(kazeevn) more elegant
    assert len(a.shape) == 2
    assert a.shape[1] == 2
    return np.cross(a, np.array((0, 0, 1)))[:, :2]/np.expand_dims(np.linalg.norm(a, axis=1), 1)


def get_momentum(velocity, mass):
    return (velocity * mass).sum(axis=0)

def get_mean_momentum_state(state):
    return (state.velocity * state.mass).mean(axis=0)


def init_at_temperature(key, positions, temeprature, init_fun, mass):
    # T [a. u.] = mean(m*v**2)/dim
    # T [K] = T [a. u.]/K_B_adjusted
    mean_m_v2 = positions.shape[1]*temeprature*K_B_adjusted
    # Positions is assumed to be an energy minimum
    # For a thermodynamic equilibrium, energies must be equally
    # distributed among the particles with different masses
    mean_v2 = np.expand_dims(mean_m_v2 / mass, 1)
    # v^2 = v_x^2+v_y^2+...
    # TODO(kazeevn) remove jit
    state = jit(init_fun)(key, positions, velocity_scale=mean_v2/positions.shape[1], mass=mass)
    v_adjusted = state.velocity
    # We set the total angular momentum around CoM at 0
    center_of_mass = np.sum(state.mass*state.position, axis=0)/np.sum(state.mass)
    position_rel_com = state.position - center_of_mass
    angular_momentum = get_angular_momentum(position_rel_com, v_adjusted, state.mass)
    com_distances = np.linalg.norm(position_rel_com, axis=1)
    points_to_adjust = (com_distances > 0.1*com_distances.max())
    p_abs_adjustment = angular_momentum/np.sum(com_distances[points_to_adjust])
    p_adjustment = get_perpendicular(position_rel_com[points_to_adjust])*p_abs_adjustment
    v_adjusted = index_add(v_adjusted,
                           points_to_adjust,
                           p_adjustment / np.expand_dims(mass[points_to_adjust], 1))
    # We set the total momentum at 0
    momentum = get_momentum(v_adjusted, state.mass)
    v_adjusted = v_adjusted - momentum/np.sum(state.mass)
    # Since we are interested in a specific temperature value, we rescale the velocities
    kinetic_energy_mean = quantity.kinetic_energy(v_adjusted, state.mass) / positions.shape[0]
    kinetic_energy_mean_target = mean_m_v2 * 0.5
    # We also multiply v by sqrt(2), as the system is currently at the energy minimum,
    # and in equlibrium potential energy \approx kinetic
    # After relaxation temeprature doesn't fully match, as the relation is approximate
    v_adjusted = v_adjusted * np.sqrt(2 * kinetic_energy_mean_target / kinetic_energy_mean)
    return simulate.NVEState(state.position, v_adjusted, state.acceleration, state.mass)






