import numpy as np
import time


def initialize_centroids(X, k, seed=42):
    """
    Initialize k centroids by randomly sampling from the dataset.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        k (int): Number of clusters.
        seed (int): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Initial centroids of shape (k, n_features).
    """
    np.random.seed(seed)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices].copy()


def assign_clusters(X, centroids):
    """
    Assign each sample to the nearest centroid using Euclidean distance.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        centroids (numpy.ndarray): Centroid matrix of shape (k, n_features).

    Returns:
        numpy.ndarray: Cluster assignments of shape (n_samples,).
    """
    # compute squared distances from each sample to each centroid
    # broadcasting: (n_samples, 1, n_features) - (k, n_features) -> (n_samples, k)
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X, assignments, k):
    """
    Recompute centroids as the mean of assigned samples.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        assignments (numpy.ndarray): Cluster assignments of shape (n_samples,).
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Updated centroids of shape (k, n_features).
    """
    centroids = np.zeros((k, X.shape[1]))
    for c in range(k):
        members = X[assignments == c]
        # keep old centroid if no samples are assigned to avoid NaN
        if len(members) > 0:
            centroids[c] = members.mean(axis=0)
    return centroids


def compute_inertia(X, assignments, centroids):
    """
    Compute within-cluster sum of squares (inertia).

    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        assignments (numpy.ndarray): Cluster assignments of shape (n_samples,).
        centroids (numpy.ndarray): Centroid matrix of shape (k, n_features).

    Returns:
        float: Total inertia across all clusters.
    """
    inertia = 0.0
    for c in range(len(centroids)):
        members = X[assignments == c]
        if len(members) > 0:
            inertia += ((members - centroids[c]) ** 2).sum()
    return inertia


def kmeans_serial(X, k, max_iter=100, tol=1e-4, seed=42):
    """
    Run K-Means clustering using the serial baseline implementation.

    Alternates between assignment and update steps until convergence
    or max_iter is reached. Convergence is detected when the maximum
    centroid shift between iterations falls below tol.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance for centroid shift.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Results containing:
            - centroids (numpy.ndarray): Final centroids.
            - assignments (numpy.ndarray): Final cluster assignments.
            - inertia (float): Final within-cluster sum of squares.
            - iterations (int): Number of iterations until convergence.
            - time_seconds (float): Total runtime.
    """
    centroids = initialize_centroids(X, k, seed)

    start = time.perf_counter()

    for iteration in range(max_iter):
        assignments = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, assignments, k)

        # check convergence — stop if centroids barely moved
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        print(f"  Iteration {iteration + 1:3d} — shift={shift:.6f}")

        if shift < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break

    elapsed = time.perf_counter() - start
    inertia = compute_inertia(X, assignments, centroids)

    return {
        "centroids": centroids,
        "assignments": assignments,
        "inertia": inertia,
        "iterations": iteration + 1,
        "time_seconds": round(elapsed, 6),
    }


if __name__ == "__main__":
    X = np.load("data/covertype.npy")

    print(f"Dataset shape: {X.shape}")
    print(f"Running serial K-Means with k=7\n")

    results = kmeans_serial(X, k=7, max_iter=50)

    print(f"\nInertia:    {results['inertia']:.2f}")
    print(f"Iterations: {results['iterations']}")
    print(f"Time:       {results['time_seconds']:.4f}s")