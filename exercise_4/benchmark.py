import numpy as np
import time
import csv
from mpi4py import MPI
from serial import kmeans_serial
from distributed import kmeans_distributed


def benchmark_serial(X, k_values, max_iter=50):
    """
    Run serial K-Means for multiple values of k and record performance.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        k_values (list[int]): List of k values to evaluate.
        max_iter (int): Maximum number of iterations per run.

    Returns:
        list[dict]: Benchmark records with k, time, inertia and iterations.
    """
    records = []
    for k in k_values:
        print(f"  Serial K-Means k={k}...")
        results = kmeans_serial(X, k=k, max_iter=max_iter)
        records.append({
            "strategy": "serial",
            "k": k,
            "num_workers": 1,
            "time_seconds": results["time_seconds"],
            "inertia": round(results["inertia"], 4),
            "iterations": results["iterations"],
            "speedup": 1.0,
        })
        print(f"    time={results['time_seconds']:.4f}s  inertia={results['inertia']:.2f}  iter={results['iterations']}")
    return records


def benchmark_distributed(k_values, max_iter=50):
    """
    Run distributed K-Means for multiple values of k and record performance.

    Must be launched with mpirun. Only root process returns results.

    Args:
        k_values (list[int]): List of k values to evaluate.
        max_iter (int): Maximum number of iterations per run.

    Returns:
        list[dict] or None: Benchmark records on root process, None elsewhere.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    records = []
    for k in k_values:
        if rank == 0:
            print(f"  Distributed K-Means k={k} workers={size}...")

        results = kmeans_distributed(k=k, max_iter=max_iter)

        if rank == 0:
            records.append({
                "strategy": "distributed",
                "k": k,
                "num_workers": size,
                "time_seconds": results["time_seconds"],
                "inertia": round(results["inertia"], 4),
                "iterations": results["iterations"],
                "speedup": None,  # filled after comparing with serial
            })
            print(f"    time={results['time_seconds']:.4f}s  inertia={results['inertia']:.2f}  iter={results['iterations']}")

    if rank == 0:
        return records
    return None


def compute_speedups(serial_records, distributed_records):
    """
    Fill speedup values by comparing serial and distributed times per k.

    Args:
        serial_records (list[dict]): Serial benchmark records.
        distributed_records (list[dict]): Distributed benchmark records.

    Returns:
        list[dict]: Distributed records with speedup values filled in.
    """
    serial_times = {r["k"]: r["time_seconds"] for r in serial_records}
    for r in distributed_records:
        if r["k"] in serial_times:
            r["speedup"] = round(serial_times[r["k"]] / r["time_seconds"], 4)
    return distributed_records


def save_results(records, path="benchmark_results.csv"):
    """
    Save benchmark results to a CSV file.

    Args:
        records (list[dict]): Combined serial and distributed records.
        path (str): Output file path.
    """
    fieldnames = ["strategy", "k", "num_workers", "time_seconds", "inertia", "iterations", "speedup"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    K_VALUES = [3, 5, 7]
    MAX_ITER = 50

    if rank == 0:
        print(f"K-Means Benchmark")
        print(f"Dataset: Covertype (581012 x 54)")
        print(f"K values: {K_VALUES}  Max iterations: {MAX_ITER}  MPI processes: {size}\n")

        X = np.load("data/covertype.npy")
        print("--- Serial ---")
        serial_records = benchmark_serial(X, K_VALUES, MAX_ITER)
    else:
        serial_records = None

    serial_records = comm.bcast(serial_records, root=0)

    print("\n--- Distributed ---")
    distributed_records = benchmark_distributed(K_VALUES, MAX_ITER)

    if rank == 0:
        distributed_records = compute_speedups(serial_records, distributed_records)
        all_records = serial_records + distributed_records
        save_results(all_records)