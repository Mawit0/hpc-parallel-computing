import numpy as np
import time
from mpi4py import MPI
from serial import initialize_centroids, compute_inertia


def assign_clusters_local(X_local, centroids):
    """
    Assign each local sample to the nearest centroid.

    Each process runs this independently on its data partition.
    No communication is required for the assignment step.

    Args:
        X_local (numpy.ndarray): Local feature matrix of shape (local_n, n_features).
        centroids (numpy.ndarray): Current centroids of shape (k, n_features).

    Returns:
        numpy.ndarray: Local cluster assignments of shape (local_n,).
    """
    distances = np.linalg.norm(X_local[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids_distributed(X_local, local_assignments, k, comm):
    """
    Recompute global centroids by aggregating local partial sums via MPI.

    Each process computes local sums and counts for each cluster.
    These are reduced across all processes using MPI Allreduce so
    every process gets the updated global centroids simultaneously.
    This avoids an extra broadcast step after the reduction.

    Args:
        X_local (numpy.ndarray): Local feature matrix of shape (local_n, n_features).
        local_assignments (numpy.ndarray): Local cluster assignments.
        k (int): Number of clusters.
        comm: MPI communicator.

    Returns:
        numpy.ndarray: Updated global centroids of shape (k, n_features).
    """
    n_features = X_local.shape[1]

    # accumulate local sums and counts per cluster
    local_sums = np.zeros((k, n_features))
    local_counts = np.zeros(k, dtype=np.int64)

    for c in range(k):
        members = X_local[local_assignments == c]
        if len(members) > 0:
            local_sums[c] = members.sum(axis=0)
            local_counts[c] = len(members)

    # reduce local sums and counts to get global values on all processes
    global_sums = np.zeros_like(local_sums)
    global_counts = np.zeros_like(local_counts)

    comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
    comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

    # compute global centroids from aggregated sums
    centroids = np.zeros((k, n_features))
    for c in range(k):
        if global_counts[c] > 0:
            centroids[c] = global_sums[c] / global_counts[c]

    return centroids


def kmeans_distributed(k, max_iter=100, tol=1e-4, seed=42):
    """
    Run K-Means clustering using MPI data partitioning.

    The dataset is split into equal row blocks, one per process.
    Each process performs local assignment independently. Centroids
    are updated globally using MPI Allreduce to aggregate partial
    cluster statistics without requiring a gather to root.

    Communication per iteration:
        - Two Allreduce calls: one for cluster sums, one for counts.
        - Cost is O(k * n_features) per iteration, independent of n_samples.

    Args:
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance for centroid shift.
        seed (int): Random seed for reproducibility.

    Returns:
        dict or None: Results on root process containing centroids,
            assignments, inertia, iterations and time. None on other processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # root loads dataset and scatters partitions
    if rank == 0:
        X = np.load("data/covertype.npy")
        n_samples = X.shape[0]
        chunk_size = n_samples // size
        chunks = [X[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
        centroids = initialize_centroids(X, k, seed)
    else:
        chunks = None
        centroids = None

    # distribute data partitions
    X_local = comm.scatter(chunks, root=0)

    # broadcast initial centroids to all processes
    centroids = comm.bcast(centroids, root=0)

    start = time.perf_counter()

    for iteration in range(max_iter):
        # each process assigns its local samples independently
        local_assignments = assign_clusters_local(X_local, centroids)

        # aggregate local statistics and update global centroids
        new_centroids = update_centroids_distributed(X_local, local_assignments, k, comm)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if rank == 0:
            print(f"  Iteration {iteration + 1:3d} — shift={shift:.6f}")

        if shift < tol:
            if rank == 0:
                print(f"  Converged at iteration {iteration + 1}")
            break

    elapsed = time.perf_counter() - start

    # gather all local assignments to root
    all_assignments = comm.gather(local_assignments, root=0)

    if rank == 0:
        assignments = np.concatenate(all_assignments)
        X_full = np.load("data/covertype.npy")
        inertia = compute_inertia(X_full, assignments, centroids)

        return {
            "centroids": centroids,
            "assignments": assignments,
            "inertia": inertia,
            "iterations": iteration + 1,
            "time_seconds": round(elapsed, 6),
            "num_workers": size,
        }

    return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    results = kmeans_distributed(k=7, max_iter=50)

    if rank == 0:
        print(f"\nInertia:    {results['inertia']:.2f}")
        print(f"Iterations: {results['iterations']}")
        print(f"Workers:    {results['num_workers']}")
        print(f"Time:       {results['time_seconds']:.4f}s")