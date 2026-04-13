import random
import numpy as np
import math
import matplotlib.pyplot as plt

#CONSTANTS
J = 1.0
h = 0.0
k_B = 1
T = 1.5
beta = 1/(k_B * T)
lattice_size = 16


def initial_state(N):
    return np.random.choice([-1, 1], size=(N, N))

def delta_E(state, i, j):
    N = len(state)
    s = state[i][j]

    nearest_neighbors_sum = (
        state[(i + 1)% N][j]
        + state[i][(j + 1)% N]
        + state[(i - 1)% N][j]
        + state[i][(j - 1)% N]
    )
    return 2.0 * s * (J * nearest_neighbors_sum + h)
            
def step(state, beta):
    N = state.shape[0]
    n_proposals = N * N
    exp_sum = 0.0

    for _ in range(n_proposals):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        dE = delta_E(state, i, j)
        exp_sum += np.exp(-beta * dE)

        # Standard Metropolis accept/reject
        if dE <= 0 or random.random() < np.exp(-beta * dE):
            state[i, j] *= -1

    return exp_sum / n_proposals # i.e distance from eq.

def simulate(state, beta, epochs):
    history = np.empty((epochs + 1, *state.shape), dtype=np.int8)
    distances = np.empty(epochs)

    history[0] = state
    for i in range(epochs):
        distances[i] = step(state, beta)
        history[i + 1] = state
    
    return history, distances

def fit_exponential(initial_value, final_value, last_step):
    return (1 / last_step) * np.log(final_value / initial_value)

def simulate_changing_T(state, initial_beta, final_beta, epochs):
    history = np.empty((epochs + 1, *state.shape), dtype=np.int8)
    distances = np.empty(epochs)
    history[0] = state

    exponential_rate = fit_exponential(initial_beta, final_beta, epochs)

    for i in range(epochs):
        beta = initial_beta * np.exp(exponential_rate * i)
        distances[i] = step(state, beta)
        history[i + 1] = state

    return history, distances