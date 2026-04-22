import time
import csv
import numpy as np
from multiprocessing import cpu_count

from serial import matrix_multiply
from parallel_rows import parallel_multiply_rows
from parallel_cols import parallel_multiply_cols
from parallel_blocks import parallel_multiply_blocks
from strassen import strassen


def generate_matrix(rows, cols):
    """
    Generate a random matrix of given dimensions.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.

    Returns:
        list[list[float]]: Random matrix of shape (rows, cols).
    """
    np.random.seed(42)
    return np.random.rand(rows, cols).tolist()


def benchmark(A, B, num_workers, size):
    """
    Run all multiplication strategies and measure their execution times.

    Each strategy is executed once and its runtime is recorded. Results
    are returned as a list of dictionaries for easy CSV export.

    Args:
        A (list[list[float]]): Left matrix.
        B (list[list[float]]): Right matrix.
        num_workers (int): Number of workers for parallel strategies.
        size (int): Matrix size label for reporting.

    Returns:
        list[dict]: List of result records with strategy, size, workers and time.
    """
    results = []

    strategies = [
        ("serial",           lambda: matrix_multiply(A, B)),
        ("parallel_rows",    lambda: parallel_multiply_rows(A, B, num_workers)),
        ("parallel_cols",    lambda: parallel_multiply_cols(A, B, num_workers)),
        ("parallel_blocks",  lambda: parallel_multiply_blocks(A, B, num_workers)),
        ("strassen",         lambda: strassen(A, B)),
    ]

    for name, fn in strategies:
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start

        results.append({
            "strategy": name,
            "size": size,
            "num_workers": num_workers,
            "time_seconds": round(elapsed, 6),
        })

        print(f"  {name:<20} size={size}  workers={num_workers}  time={elapsed:.4f}s")

    return results


def save_results(records, path="results.csv"):
    """
    Save benchmark results to a CSV file.

    Args:
        records (list[dict]): List of result records.
        path (str): Output file path.
    """
    fieldnames = ["strategy", "size", "num_workers", "time_seconds"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    # matrix sizes to evaluate — increase gradually to observe scaling behavior
    sizes = [64, 128, 256, 512]
    num_workers = 4  # adjust based on available CPU cores

    print(f"Available CPU cores: {cpu_count()}")
    print(f"Running benchmarks with {num_workers} workers\n")

    all_results = []

    for size in sizes:
        print(f"--- Size {size}x{size} ---")
        A = generate_matrix(size, size)
        B = generate_matrix(size, size)
        records = benchmark(A, B, num_workers, size)
        all_results.extend(records)

    save_results(all_results)