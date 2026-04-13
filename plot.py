import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import mc

lattice_size = 128

lattice = mc.initial_state(lattice_size)
distances_by_T = []
for T in range(1, 4):
    history, distances = mc.simulate(lattice, 1/T, 700)
    distances_by_T.append((distances, T))

    fig, ax = plt.subplots()
    im = ax.imshow(lattice, cmap='bwr', interpolation='nearest')
    ax.legend(
        handles=[
            mpatches.Patch(color='red', label='Spin +1'),
            mpatches.Patch(color='blue', label='Spin −1'),
        ],
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    def update(frame):
        new_data = history[frame]
        im.set_array(new_data)
        return [im]

    ani = FuncAnimation(fig, update, frames=len(history), interval=75, blit=True)
    ani.save(f"history_T={T}.mp4")

#Plot Creutz estimator
for distances, T in distances_by_T:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(np.arange(0, len(distances), 1), distances)
    ax.set_ylim(0.5, 1.5)  # adjust to your actual data range
    ax.set_xlabel("Step")
    ax.set_ylabel("Creutz estimator")
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5)
    fig.savefig(f"plots/creutz_est_T={T}.png", bbox_inches='tight')