import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import csv


def load_sparse_matrix(path):
    """
    Load a sparse matrix from a Matrix Market file (.mtx).

    Matrix Market is the standard format used by the SuiteSparse
    Matrix Collection. Downloaded files can be loaded directly
    with scipy.io.mmread.

    Args:
        path (str): Path to the .mtx file.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix in CSR format.
    """
    from scipy.io import mmread
    # CSR format is used because it allows efficient row slicing
    # which is consistent with the row-based partition strategy
    return csr_matrix(mmread(path))


def sparse_serial(A, B):
    """
    Compute sparse matrix product C = A x B using scipy.

    This serves as the baseline for sparse matrix experiments.
    scipy handles the sparsity pattern internally and avoids
    computing zero-valued products.

    Args:
        A (scipy.sparse.csr_matrix): Left sparse matrix.
        B (scipy.sparse.csr_matrix): Right sparse matrix.

    Returns:
        scipy.sparse.csr_matrix: Result sparse matrix C.
    """
    return A.dot(B)


def _sparse_worker(args):
    """
    Worker function for sparse parallel row multiplication.

    Must be defined at module level so multiprocessing can pickle it.

    Args:
        args (tuple): A tuple containing:
            - A_block (scipy.sparse.csr_matrix): Row slice of A.
            - B (scipy.sparse.csr_matrix): Full matrix B.

    Returns:
        scipy.sparse.csr_matrix: Partial result for this row block.
    """
    A_block, B = args
    return A_block.dot(B)


def sparse_parallel_rows(A, B, num_workers):
    """
    Compute sparse matrix product using row-based partitioning with multiprocessing.

    A is split into row slices. Each worker computes its slice multiplied
    by the full B. Results are stacked vertically to form C.

    Args:
        A (scipy.sparse.csr_matrix): Left sparse matrix.
        B (scipy.sparse.csr_matrix): Right sparse matrix.
        num_workers (int): Number of parallel worker processes.

    Returns:
        scipy.sparse.csr_matrix: Result sparse matrix C.
    """
    from multiprocessing import Pool

    chunk_size = A.shape[0] // num_workers
    chunks = [(A[i * chunk_size:(i + 1) * chunk_size], B) for i in range(num_workers)]

    with Pool(processes=num_workers) as pool:
        results = pool.map(_sparse_worker, chunks)

    return sp.vstack(results)


def benchmark_sparse(path_A, path_B, num_workers=4):
    """
    Run serial and parallel benchmarks on two sparse matrices.

    Loads two sparse matrices from disk, runs both strategies,
    measures execution times and reports speedup.

    Args:
        path_A (str): Path to the first sparse matrix .mtx file.
        path_B (str): Path to the second sparse matrix .mtx file.
        num_workers (int): Number of workers for parallel strategy.

    Returns:
        list[dict]: Benchmark records with strategy, shape and time.
    """
    print("Loading sparse matrices...")
    A = load_sparse_matrix(path_A)
    B = load_sparse_matrix(path_B)

    # ensure dimensions are compatible for multiplication
    if A.shape[1] != B.shape[0]:
        B = B.T
        print("Transposed B to make dimensions compatible.")

    print(f"A shape: {A.shape}  nnz: {A.nnz}  density: {A.nnz / (A.shape[0] * A.shape[1]):.4f}")
    print(f"B shape: {B.shape}  nnz: {B.nnz}  density: {B.nnz / (B.shape[0] * B.shape[1]):.4f}")

    results = []

    # serial benchmark
    start = time.perf_counter()
    sparse_serial(A, B)
    serial_time = time.perf_counter() - start
    print(f"\n  serial          time={serial_time:.4f}s")
    results.append({"strategy": "sparse_serial", "shape": str(A.shape), "time_seconds": round(serial_time, 6)})

    # parallel benchmark
    start = time.perf_counter()
    sparse_parallel_rows(A, B, num_workers)
    parallel_time = time.perf_counter() - start
    print(f"  parallel_rows   time={parallel_time:.4f}s")
    results.append({"strategy": "sparse_parallel_rows", "shape": str(A.shape), "time_seconds": round(parallel_time, 6)})

    speedup = serial_time / parallel_time
    print(f"\n  Speedup: {speedup:.2f}x with {num_workers} workers")

    return results


def save_results(records, path="sparse_results.csv"):
    """
    Save sparse benchmark results to a CSV file.

    Args:
        records (list[dict]): List of result records.
        path (str): Output file path.
    """
    fieldnames = ["strategy", "shape", "time_seconds"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    path_A = "data/west0067/west0067.mtx"
    path_B = "data/west0067/west0067.mtx"

    records = benchmark_sparse(path_A, path_B, num_workers=4)
    save_results(records)