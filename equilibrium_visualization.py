import mc
import plot

lattice_size = 128

temps = [1e-4, 1e-1, 1, 2, 2.1, 2.2, 2.3, 3, 4, 5]

for T in temps:
    print(f"Starting simulation for T = {T}")

    lattice = mc.initial_state(lattice_size)
    history_hb, energy_hb, all_m_hb = mc.simulate_heat_bath(
        state=lattice, beta=1 / T, epochs=10000
    )

    lattice = mc.initial_state(lattice_size)
    history_met, energy_met, all_m_met = mc.simulate_metropolis(
        state=lattice, beta=1 / T, epochs=10000
    )

    plot.generate_video_fast(history_met[::20], f"metropolis_T={T}", lattice_size, T)
    plot.generate_video_fast(history_hb[::20], f"heat_bath_T={T}", lattice_size, T)

    plot.save_energy_plot(energy_met, T, f"metropolis_avg_E_T={T}", lattice_size)
    plot.save_energy_plot(energy_hb, T, f"heat_bath_avg_E_T={T}", lattice_size)
