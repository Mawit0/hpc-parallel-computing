import numpy as np


# cell states
EMPTY = 0        # non-burnable or outside valid domain
SUSCEPTIBLE = 1  # vegetation that can catch fire
BURNING = 2      # actively burning cell
BURNED = 3       # already consumed, cannot burn again

# default simulation parameters
IGNITION_PROBABILITY = 0.3   # base probability of catching fire from a burning neighbor
BURN_LIFETIME = 3            # number of steps a cell stays in BURNING state before becoming BURNED


def initialize_grid(grid_size, ignition_grid):
    """
    Initialize the automaton state grid from NASA FIRMS hotspot data.

    All cells start as SUSCEPTIBLE vegetation. Cells that correspond
    to hotspot detections in the ignition grid are set to BURNING,
    representing the initial fire front derived from satellite data.

    Args:
        grid_size (int): Number of cells per dimension.
        ignition_grid (numpy.ndarray): Binary grid from data.py where
            1 indicates a detected hotspot.

    Returns:
        tuple:
            - state (numpy.ndarray): Initial state grid of shape (grid_size, grid_size).
            - lifetime (numpy.ndarray): Burn lifetime counter per cell.
    """
    state = np.full((grid_size, grid_size), SUSCEPTIBLE, dtype=int)
    lifetime = np.zeros((grid_size, grid_size), dtype=int)

    # set hotspot cells as initial ignition points
    state[ignition_grid == 1] = BURNING
    lifetime[ignition_grid == 1] = BURN_LIFETIME

    return state, lifetime


def step(state, lifetime, frp_grid=None):
    """
    Advance the automaton by one time step.

    Transition rules:
        - BURNING cells decrement their lifetime counter. When lifetime
          reaches 0 they transition to BURNED.
        - SUSCEPTIBLE cells catch fire if at least one neighbor is BURNING.
          The ignition probability scales with FRP if frp_grid is provided.
        - BURNED and EMPTY cells never change state.

    Neighborhood: 4-connected (up, down, left, right).

    Args:
        state (numpy.ndarray): Current state grid.
        lifetime (numpy.ndarray): Burn lifetime counter grid.
        frp_grid (numpy.ndarray or None): Fire radiative power per cell.
            If provided, scales ignition probability locally.

    Returns:
        tuple:
            - new_state (numpy.ndarray): Updated state grid.
            - new_lifetime (numpy.ndarray): Updated lifetime grid.
    """
    new_state = state.copy()
    new_lifetime = lifetime.copy()
    rows, cols = state.shape

    for i in range(rows):
        for j in range(cols):

            if state[i][j] == BURNING:
                # burning cells consume fuel each step
                new_lifetime[i][j] -= 1
                if new_lifetime[i][j] <= 0:
                    new_state[i][j] = BURNED

            elif state[i][j] == SUSCEPTIBLE:
                # check 4-connected neighbors for burning cells
                neighbors = []
                if i > 0:          neighbors.append(state[i-1][j])
                if i < rows - 1:   neighbors.append(state[i+1][j])
                if j > 0:          neighbors.append(state[i][j-1])
                if j < cols - 1:   neighbors.append(state[i][j+1])

                burning_neighbors = neighbors.count(BURNING)

                if burning_neighbors > 0:
                    # scale probability by FRP if available
                    frp_factor = 1.0
                    if frp_grid is not None and frp_grid[i][j] > 0:
                        frp_factor = min(2.0, 1.0 + frp_grid[i][j] / 100.0)

                    prob = min(1.0, IGNITION_PROBABILITY * burning_neighbors * frp_factor)

                    if np.random.random() < prob:
                        new_state[i][j] = BURNING
                        new_lifetime[i][j] = BURN_LIFETIME

    return new_state, new_lifetime


def run_simulation(state, lifetime, frp_grid=None, num_steps=20):
    """
    Run the serial forest fire automaton for a given number of steps.

    At each step the transition rules are applied to the entire grid.
    Snapshots of the state are saved at every step for visualization.

    Args:
        state (numpy.ndarray): Initial state grid from initialize_grid.
        lifetime (numpy.ndarray): Initial lifetime grid from initialize_grid.
        frp_grid (numpy.ndarray or None): FRP grid from data.py.
        num_steps (int): Number of simulation steps to run.

    Returns:
        list[numpy.ndarray]: List of state snapshots, one per step.
    """
    snapshots = [state.copy()]

    for step_idx in range(num_steps):
        state, lifetime = step(state, lifetime, frp_grid)
        snapshots.append(state.copy())

        burning = (state == BURNING).sum()
        burned = (state == BURNED).sum()
        print(f"  Step {step_idx + 1:3d} — burning: {burning:5d}  burned: {burned:5d}")

        # stop early if no burning cells remain
        if burning == 0:
            print("  Fire extinguished.")
            break

    return snapshots


if __name__ == "__main__":
    np.random.seed(42)

    # load grids generated by data.py
    ignition_grid = np.load("data/grid.npy")
    frp_grid = np.load("data/frp_grid.npy")

    grid_size = ignition_grid.shape[0]
    state, lifetime = initialize_grid(grid_size, ignition_grid)

    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Initial burning cells: {(state == BURNING).sum()}")
    print(f"Running simulation...\n")

    snapshots = run_simulation(state, lifetime, frp_grid, num_steps=20)
    np.save("data/snapshots.npy", np.array(snapshots))

    print(f"\nSimulation complete. {len(snapshots)} snapshots saved.")