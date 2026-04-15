import numpy as np
import numba

# CONSTANTS
J = 1.0
h = 0.0 # Magnetic Field
k_B = 1
T = 1.5
beta = 1 / (k_B * T)


def initial_state(N):
    return np.random.choice([-1, 1], size=(N, N))


@numba.jit
def energy_per_spin(state):
    N = state.shape[0]
    E = 0.0
    # Sum over every pair
    for i in range(N):
        for j in range(N):
            E += -J * state[i, j] * (state[(i + 1) % N, j] + state[i, (j + 1) % N])
            E += -h * state[i, j]
    return E / (N * N)

@numba.jit
def sum_over_spins(state):
    N = state.shape[0]
    s = 0
    for i in range(N):
        for j in range(N):
            s += state[i, j]
    return s/(N*N)

@numba.jit
def nearest_neighbors_sum(state, i, j):
    N = len(state)
    return (
        state[(i + 1) % N][j]
        + state[i][(j + 1) % N]
        + state[(i - 1) % N][j]
        + state[i][(j - 1) % N]
    )

@numba.jit
def step_heat_bath(state, beta):
    for _ in range(len(state)**2):
        i, j = np.random.choice(len(state)), np.random.choice(len(state))
        nn = nearest_neighbors_sum(state, i, j)

        p1, p2 = np.exp(beta * nn), np.exp(-beta * nn)
        Z = p1+p2
        p = p1/Z

        if np.random.random() < p:
            state[i, j] = 1
        else:
            state[i, j] = -1

@numba.jit
def step_metropolis(state, beta):
    for _ in range(len(state)**2):
        i, j = np.random.choice(len(state)), np.random.choice(len(state))
        nn = nearest_neighbors_sum(state, i, j)
        dE = 2.0 * state[i, j] * (J * nn + h)

        # Standard Metropolis accept/reject
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            state[i, j] *= -1


def simulate_metropolis(state, beta, epochs):
    history = np.empty((epochs + 1, *state.shape), dtype=np.int8)
    energies = np.empty(epochs)
    history[0] = state

    for i in range(epochs):
        step_metropolis(state, beta)
        history[i + 1] = state
        energies[i] = energy_per_spin(state)
    return history, energies

def simulate_heat_bath(state, beta, epochs):
    history = np.empty((epochs + 1, *state.shape), dtype=np.int8)
    energies = np.empty(epochs)
    history[0] = state

    for i in range(epochs):
        step_heat_bath(state, beta)
        history[i + 1] = state
        energies[i] = energy_per_spin(state)
    return history, energies

def fit_exponential(initial_value, final_value, last_step):
    return (1 / last_step) * np.log(final_value / initial_value)


def simulate_changing_T_heat_bath(state, initial_beta: float, final_beta: float, epochs: int, linear: bool):
    history = np.empty((epochs + 1, *state.shape), dtype=np.int8)
    energies = np.empty(epochs)
    history[0] = state

    if linear:
        slope = (final_beta - initial_beta) / epochs
        beta = lambda i: initial_beta + (i * slope)
    else:
        exponential_rate = fit_exponential(initial_beta, final_beta, epochs)
        beta = lambda i: initial_beta * np.exp(exponential_rate * i)

    for i in range(epochs):
        step_heat_bath(state, beta(i))
        history[i + 1] = state
        energies[i] = energy_per_spin(state)

    return history, energies
