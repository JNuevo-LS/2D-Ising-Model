import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import mc

lattice_size = 256

energies_by_T = []
for T in (0.5, 1, 2, 3, 4):
    lattice = mc.initial_state(lattice_size)
    history, energies = mc.simulate(lattice, 1 / T, 100)
    energies_by_T.append((energies, T))

    fig, ax = plt.subplots()
    im = ax.imshow(lattice, cmap="bwr", interpolation="nearest")
    ax.set_title(
        f"{lattice_size}x{lattice_size} Lattice at T = {T}\u00b0K (Natural Units)"
    )
    ax.legend(
        handles=[
            mpatches.Patch(color="red", label="Spin +1"),
            mpatches.Patch(color="blue", label="Spin −1"),
        ],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    def update(frame):
        new_data = history[frame]
        im.set_array(new_data)
        return [im]

    ani = FuncAnimation(fig, update, frames=len(history), interval=75, blit=True)
    ani.save(f"videos/history_T={T}.mp4")

for energies, T in energies_by_T:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(np.arange(0, len(energies), 1), energies)
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Energy")
    ax.set_title(
        f"{lattice_size}x{lattice_size} Lattice at T = {T}\u00b0K (Natural Units)"
    )
    fig.savefig(f"plots/average_energy_T={T}.png", bbox_inches="tight")
    plt.close()
