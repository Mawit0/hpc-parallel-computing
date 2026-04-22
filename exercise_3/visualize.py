import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio
import os

from automaton import EMPTY, SUSCEPTIBLE, BURNING, BURNED

# color map matching cell states
# 0=EMPTY: white, 1=SUSCEPTIBLE: green, 2=BURNING: red, 3=BURNED: black
CMAP = mcolors.ListedColormap(["white", "green", "red", "black"])
NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], CMAP.N)


def plot_snapshot(state, step_idx, output_dir="data/snapshots"):
    """
    Save a single snapshot of the automaton state as a PNG image.

    Args:
        state (numpy.ndarray): State grid at a given time step.
        step_idx (int): Step index used for filename and title.
        output_dir (str): Directory where PNG files are saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(state, cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_title(f"Forest fire — step {step_idx}", fontsize=12)
    ax.axis("off")

    # add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="white",  ec="gray", label="Empty"),
        plt.Rectangle((0, 0), 1, 1, color="green",  label="Susceptible"),
        plt.Rectangle((0, 0), 1, 1, color="red",    label="Burning"),
        plt.Rectangle((0, 0), 1, 1, color="black",  label="Burned"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    path = os.path.join(output_dir, f"step_{step_idx:03d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_all_snapshots(snapshots, output_dir="data/snapshots"):
    """
    Save PNG snapshots for all simulation steps.

    Args:
        snapshots (list[numpy.ndarray]): List of state grids from run_simulation.
        output_dir (str): Directory where PNG files are saved.
    """
    print(f"Saving {len(snapshots)} snapshots to {output_dir}...")
    for i, state in enumerate(snapshots):
        plot_snapshot(state, i, output_dir)
    print("Snapshots saved.")


def create_animation(snapshot_dir="data/snapshots", output_path="data/fire_animation.gif", fps=3):
    """
    Create an animated GIF from saved snapshot PNG files.

    Args:
        snapshot_dir (str): Directory containing snapshot PNG files.
        output_path (str): Output path for the GIF file.
        fps (int): Frames per second for the animation.
    """
    frames = sorted([
        os.path.join(snapshot_dir, f)
        for f in os.listdir(snapshot_dir)
        if f.endswith(".png")
    ])

    if not frames:
        print("No snapshots found. Run plot_all_snapshots first.")
        return

    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(output_path, images, fps=fps)
    print(f"Animation saved to {output_path}")


def plot_burn_progress(snapshots, output_path="data/burn_progress.png"):
    """
    Plot the number of burning and burned cells over time.

    Args:
        snapshots (list[numpy.ndarray]): List of state grids from run_simulation.
        output_path (str): Output path for the plot.
    """
    burning_counts = [(s == BURNING).sum() for s in snapshots]
    burned_counts  = [(s == BURNED).sum()  for s in snapshots]
    steps = list(range(len(snapshots)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, burning_counts, color="red",   label="Burning cells")
    ax.plot(steps, burned_counts,  color="black",  label="Burned cells")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Number of cells")
    ax.set_title("Fire propagation over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Burn progress plot saved to {output_path}")


if __name__ == "__main__":
    snapshots_path = "data/snapshots.npy"

    if not os.path.exists(snapshots_path):
        print("snapshots.npy not found. Run automaton.py first.")
        exit(1)

    snapshots = list(np.load(snapshots_path, allow_pickle=False))

    plot_all_snapshots(snapshots)
    create_animation()
    plot_burn_progress(snapshots)

    print("\nVisualization complete.")