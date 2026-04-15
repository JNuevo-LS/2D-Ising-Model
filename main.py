import mc
import plot

lattice_size = 128
energies_met_all, energies_hb_all = [], []
for T in (1e-1, 1e-2, 1e-3, 1e-4):
    print(f"Starting simulation for T = {T}")

    lattice = mc.initial_state(lattice_size)
    history_met, energies_met = mc.simulate_metropolis(state = lattice, beta = 1 / T, epochs = 5000)
    energies_met_all.append((energies_met, T))

    lattice = mc.initial_state(lattice_size)
    history_hb, energies_hb = mc.simulate_heat_bath(state = lattice, beta = 1 / T, epochs = 5000)
    energies_hb_all.append((energies_hb, T))

    plot.generate_video_fast(history_met[::10], f"metropolis_T={T}", lattice_size, T)
    plot.generate_video_fast(history_hb[::10], f"heat_bath_T={T}", lattice_size, T)

# for energies, T in energies_met_all:
#     plot.save_energy_plot(energies, T, f"metropolis_avg_E_T={T}")

# for energies, T in energies_hb_all:
#     plot.save_energy_plot(energies, T, f"heat_bath_avg_E_T={T}")
