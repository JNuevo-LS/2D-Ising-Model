import random
import numpy as np
import numba

# CONSTANTS
J = 1.0
h = 0.0
k_B = 1
T = 1.5
beta = 1 / (k_B * T)


def initial_state(N):
    return np.random.choice([-1, 1], size=(N, N))


@numba.jit
def energy_per_spin(state):
    N = state.shape[0]
    E = 0.0
    for i in range(N):
        for j in range(N):
            E += -J * state[i, j] * (state[(i + 1) % N, j] + state[i, (j + 1) % N])
            E += -h * state[i, j]
    return E / (N * N)


@numba.jit
def step(state, beta):
    N = state.shape[0]
    exp_sum = 0.0

    def delta_E(state, i, j) -> float:
        N = len(state)
        s = state[i][j]

        nearest_neighbors_sum = (
            state[(i + 1) % N][j]
            + state[i][(j + 1) % N]
            + state[(i - 1) % N][j]
            + state[i][(j - 1) % N]
        )
        return 2.0 * s * (J * nearest_neighbors_sum + h)

    for i in range(len(state)):
        for j in range(len(state[0])):
            dE: float = delta_E(state, i, j)
            exp_sum += np.exp(-beta * dE)

            # Standard Metropolis accept/reject
            if dE <= 0 or random.random() < np.exp(-beta * dE):
                state[i, j] *= -1


def simulate(state, beta, epochs):
    history = np.empty((epochs + 1, *state.shape), dtype=np.int8)
    energies = np.empty(epochs)
    history[0] = state

    for i in range(epochs):
        step(state, beta)
        history[i + 1] = state
        energies[i] = energy_per_spin(state)
    return history, energies


def fit_exponential(initial_value, final_value, last_step):
    return (1 / last_step) * np.log(final_value / initial_value)


def simulate_changing_T(state, initial_beta, final_beta, epochs):
    history = np.empty((epochs + 1, *state.shape), dtype=np.int8)
    distances = np.empty(epochs)
    energies = np.empty(epochs)
    history[0] = state

    exponential_rate = fit_exponential(initial_beta, final_beta, epochs)

    for i in range(epochs):
        beta = initial_beta * np.exp(exponential_rate * i)
        step(state, beta)
        history[i + 1] = state
        energies[i] = energy_per_spin(state)

    return history, distances
