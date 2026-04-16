import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


def generate_video_fast(history, filename, lattice_size, T, scale=6):
    os.makedirs("videos", exist_ok=True)

    title = f"{lattice_size}x{lattice_size} Lattice at T = {T} (Natural Units)"

    WIDTH, HEIGHT = 1920, 1072
    TITLE_BAR_HEIGHT = 50  # space for title

    available_height = HEIGHT - TITLE_BAR_HEIGHT
    scale = min(scale, available_height // lattice_size, WIDTH // lattice_size)

    with imageio.get_writer(
        f"videos/{filename}.mp4", fps=30, codec="libx264"
    ) as writer:
        for step, frame in enumerate(history):
            img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            img[frame == 1] = [255, 0, 0]  # red
            img[frame == -1] = [0, 0, 255]  # blue

            # Scale lattice
            img = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))

            h, w, _ = img.shape

            canvas = (
                np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
            )  # white background

            # Compute Lattice centered position3,
            available_height = HEIGHT - TITLE_BAR_HEIGHT
            y_offset = TITLE_BAR_HEIGHT + (available_height - h) // 2
            x_offset = (WIDTH - w) // 2

            # Place lattice
            canvas[y_offset : y_offset + h, x_offset : x_offset + w] = img

            # Title background bar
            cv2.rectangle(
                canvas, (0, 0), (WIDTH, TITLE_BAR_HEIGHT), (255, 255, 255), -1
            )

            # Title text centering
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)

            text_x = (WIDTH - text_w) // 2
            text_y = (TITLE_BAR_HEIGHT + text_h) // 2

            cv2.putText(
                canvas,
                title,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),  # black text
                thickness,
                cv2.LINE_AA,
            )

            writer.append_data(canvas)


def save_energy_plot(energies, T, filename, lattice_size):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(np.arange(0, len(energies), 1), energies)
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Energy")
    ax.set_title(
        f"{lattice_size}x{lattice_size} Lattice at T = {T}\u00b0K (Na[],tural Units)"
    )
    fig.savefig(f"plots/{filename}.png", bbox_inches="tight")
    plt.close()


def plot_m_histogram(m, T):
    plt.hist(m)
    plt.title(f"Magnetization at T={T}")
    plt.savefig(f"plots/m_at_T={T}.png")
    plt.close()


def overlay_m_histograms(mag_by_temp: dict):
    temps = sorted(mag_by_temp.keys(), reverse=True)
    fig, axes = plt.subplots(len(temps), 1, figsize=(6, 2 * len(temps)), sharex=True)
    for ax, T in zip(axes, temps):
        bins = np.linspace(-1, 1, 101)
        ax.hist(mag_by_temp[T], bins=bins, density=True, alpha=0.7)
        ax.set_ylabel("P(m)")
        ax.set_title(f"T = {T}")
        ax.set_xlim(-1.1, 1.1)
    axes[-1].set_xlabel("m")
    fig.suptitle("Magnetization Distribution by Temperature", y=1.01)
    plt.tight_layout()
    plt.savefig("plots/magnetization_by_temp.png", dpi=150, bbox_inches="tight")
    plt.close()
