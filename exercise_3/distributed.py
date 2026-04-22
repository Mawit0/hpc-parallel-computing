import numpy as np
from mpi4py import MPI
from automaton import SUSCEPTIBLE, BURNING, BURNED, BURN_LIFETIME, IGNITION_PROBABILITY


def step_distributed(local_state, local_lifetime, top_ghost, bottom_ghost, frp_grid=None):
    """
    Advance one time step on a local subdomain with ghost row exchange.

    Each process owns a horizontal slice of the grid. To correctly compute
    transitions at subdomain boundaries, ghost rows are exchanged with
    neighboring processes before each step. This ensures that border cells
    see their actual neighbors even when those neighbors belong to another process.

    Args:
        local_state (numpy.ndarray): Local subdomain state grid.
        local_lifetime (numpy.ndarray): Local lifetime counter grid.
        top_ghost (numpy.ndarray or None): Ghost row from the process above.
        bottom_ghost (numpy.ndarray or None): Ghost row from the process below.
        frp_grid (numpy.ndarray or None): Local FRP grid slice.

    Returns:
        tuple:
            - new_state (numpy.ndarray): Updated local state grid.
            - new_lifetime (numpy.ndarray): Updated local lifetime grid.
    """
    new_state = local_state.copy()
    new_lifetime = local_lifetime.copy()
    rows, cols = local_state.shape

    for i in range(rows):
        for j in range(cols):

            if local_state[i][j] == BURNING:
                new_lifetime[i][j] -= 1
                if new_lifetime[i][j] <= 0:
                    new_state[i][j] = BURNED

            elif local_state[i][j] == SUSCEPTIBLE:
                neighbors = []

                # use ghost rows at subdomain boundaries
                if i == 0 and top_ghost is not None:
                    neighbors.append(top_ghost[j])
                elif i > 0:
                    neighbors.append(local_state[i-1][j])

                if i == rows - 1 and bottom_ghost is not None:
                    neighbors.append(bottom_ghost[j])
                elif i < rows - 1:
                    neighbors.append(local_state[i+1][j])

                if j > 0:   neighbors.append(local_state[i][j-1])
                if j < cols - 1: neighbors.append(local_state[i][j+1])

                burning_neighbors = neighbors.count(BURNING)

                if burning_neighbors > 0:
                    frp_factor = 1.0
                    if frp_grid is not None and frp_grid[i][j] > 0:
                        frp_factor = min(2.0, 1.0 + frp_grid[i][j] / 100.0)

                    prob = min(1.0, IGNITION_PROBABILITY * burning_neighbors * frp_factor)

                    if np.random.random() < prob:
                        new_state[i][j] = BURNING
                        new_lifetime[i][j] = BURN_LIFETIME

    return new_state, new_lifetime


def exchange_ghost_rows(comm, local_state, rank, size):
    """
    Exchange boundary rows with neighboring MPI processes.

    Each process sends its top row to the process above and its bottom
    row to the process below. In return it receives ghost rows that
    represent the boundary of neighboring subdomains.

    Args:
        comm: MPI communicator.
        local_state (numpy.ndarray): Local subdomain state grid.
        rank (int): Rank of the current process.
        size (int): Total number of processes.

    Returns:
        tuple:
            - top_ghost (numpy.ndarray or None): Row received from process above.
            - bottom_ghost (numpy.ndarray or None): Row received from process below.
    """
    top_ghost = None
    bottom_ghost = None

    # send bottom row down, receive top ghost from above
    if rank > 0:
        comm.send(local_state[0], dest=rank - 1, tag=1)
    if rank < size - 1:
        bottom_ghost = comm.recv(source=rank + 1, tag=1)

    # send top row up, receive bottom ghost from below
    if rank < size - 1:
        comm.send(local_state[-1], dest=rank + 1, tag=2)
    if rank > 0:
        top_ghost = comm.recv(source=rank - 1, tag=2)

    return top_ghost, bottom_ghost


def run_distributed(num_steps=20):
    """
    Run the forest fire automaton using MPI domain decomposition.

    The grid is split horizontally into equal row slices, one per process.
    At each step, ghost rows are exchanged between neighboring processes
    to handle boundary transitions correctly. The root process assembles
    the full grid for reporting after each step.

    Domain decomposition strategy:
        - Grid is partitioned by rows (horizontal slices)
        - Each process handles chunk_size rows independently
        - Only boundary rows require communication (ghost exchange)
        - Communication cost is O(cols) per step, independent of grid size

    Args:
        num_steps (int): Number of simulation steps to run.

    Returns:
        list[numpy.ndarray] or None: Snapshots on root process, None elsewhere.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    np.random.seed(42)

    # root loads and distributes the grid
    if rank == 0:
        ignition_grid = np.load("data/grid.npy")
        frp_grid = np.load("data/frp_grid.npy")
        grid_size = ignition_grid.shape[0]

        from automaton import initialize_grid
        state, lifetime = initialize_grid(grid_size, ignition_grid)

        chunk_size = grid_size // size
        state_chunks = [state[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
        lifetime_chunks = [lifetime[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
        frp_chunks = [frp_grid[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
    else:
        state_chunks = None
        lifetime_chunks = None
        frp_chunks = None

    # distribute subdomains to each process
    local_state = comm.scatter(state_chunks, root=0)
    local_lifetime = comm.scatter(lifetime_chunks, root=0)
    local_frp = comm.scatter(frp_chunks, root=0)

    snapshots = []

    for step_idx in range(num_steps):
        # exchange ghost rows before computing transitions
        top_ghost, bottom_ghost = exchange_ghost_rows(comm, local_state, rank, size)

        local_state, local_lifetime = step_distributed(
            local_state, local_lifetime, top_ghost, bottom_ghost, local_frp
        )

        # gather full grid at root for reporting
        full_state = comm.gather(local_state, root=0)

        if rank == 0:
            full_grid = np.vstack(full_state)
            snapshots.append(full_grid.copy())
            burning = (full_grid == BURNING).sum()
            burned = (full_grid == BURNED).sum()
            print(f"  Step {step_idx + 1:3d} — burning: {burning:5d}  burned: {burned:5d}")
            if burning == 0:
                print("  Fire extinguished.")
                break

        # broadcast stop signal if fire is extinguished
        stop = comm.bcast((rank == 0 and burning == 0) if rank == 0 else None, root=0)
        if stop:
            break

    if rank == 0:
        np.save("data/snapshots_distributed.npy", np.array(snapshots))
        print(f"\nDistributed simulation complete. {len(snapshots)} snapshots saved.")
        return snapshots

    return None


if __name__ == "__main__":
    run_distributed(num_steps=20)