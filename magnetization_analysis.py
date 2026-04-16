import mc
import plot
import numpy as np

lattice_size = 256

mag_by_temp = {}

temps = [0.001, 1, 2, 2.1, 2.2, 2.3, 3, 4, 5]

for T in temps:
    equilibrium_time = 50000
    data_time = 5000

    # Equilibrate Lattice
    lattice = mc.initial_state(lattice_size)
    print(f"T={T}, Equilibrating lattice over {equilibrium_time} sweeps")
    history, _, _ = mc.simulate_metropolis(
        state=lattice, beta=1 / T, epochs=equilibrium_time
    )
    print(f"Lattice equilibrium time done")

    # Actually get data
    print(f"Gathering samples of m over {data_time} sweeps")
    _, _, all_m = mc.simulate_metropolis(state=lattice, beta=1 / T, epochs=data_time)

    mag_by_temp[T] = all_m
    print(f"Generating video of equilibrium history")
    plot.generate_video_fast(history[::100], f"magnetization_T={T}", lattice_size, T)

plot.overlay_m_histograms(mag_by_temp)
