import time
import numpy as np
from mpi4py import MPI


def benchmark_serial(num_steps=20):
    """
    Run and time the serial forest fire simulation.

    Args:
        num_steps (int): Number of simulation steps.

    Returns:
        float: Total execution time in seconds.
    """
    from automaton import initialize_grid, run_simulation

    ignition_grid = np.load("data/grid.npy")
    frp_grid = np.load("data/frp_grid.npy")

    np.random.seed(42)
    state, lifetime = initialize_grid(ignition_grid.shape[0], ignition_grid)

    start = time.perf_counter()
    run_simulation(state, lifetime, frp_grid, num_steps=num_steps)
    elapsed = time.perf_counter() - start

    print(f"  serial  steps={num_steps}  time={elapsed:.4f}s")
    return elapsed


def benchmark_distributed(num_steps=20):
    """
    Run and time the distributed MPI forest fire simulation.

    Must be launched with mpirun. Only the root process returns
    the elapsed time — other processes return None.

    Args:
        num_steps (int): Number of simulation steps.

    Returns:
        float or None: Execution time on root process, None elsewhere.
    """
    from distributed import run_distributed

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = time.perf_counter()
    run_distributed(num_steps=num_steps)
    elapsed = time.perf_counter() - start

    if rank == 0:
        print(f"  distributed  steps={num_steps}  workers={size}  time={elapsed:.4f}s")
        return elapsed

    return None


def save_results(serial_time, distributed_time, num_workers, num_steps, path="benchmark_results.csv"):
    """
    Save benchmark results to a CSV file.

    Args:
        serial_time (float): Serial execution time.
        distributed_time (float): Distributed execution time.
        num_workers (int): Number of MPI processes used.
        num_steps (int): Number of simulation steps.
        path (str): Output file path.
    """
    import csv
    speedup = round(serial_time / distributed_time, 4)
    records = [
        {"strategy": "serial",      "num_workers": 1,           "num_steps": num_steps, "time_seconds": round(serial_time, 6),      "speedup": 1.0},
        {"strategy": "distributed", "num_workers": num_workers,  "num_steps": num_steps, "time_seconds": round(distributed_time, 6), "speedup": speedup},
    ]
    fieldnames = ["strategy", "num_workers", "num_steps", "time_seconds", "speedup"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults saved to {path}")
    print(f"Speedup: {speedup:.2f}x with {num_workers} workers")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    NUM_STEPS = 20

    if rank == 0:
        print(f"Benchmarking forest fire automaton")
        print(f"Grid size: 100x100  Steps: {NUM_STEPS}  MPI processes: {size}\n")
        serial_time = benchmark_serial(num_steps=NUM_STEPS)
    else:
        serial_time = None

    serial_time = comm.bcast(serial_time, root=0)

    distributed_time = benchmark_distributed(num_steps=NUM_STEPS)

    if rank == 0:
        save_results(serial_time, distributed_time, size, NUM_STEPS)