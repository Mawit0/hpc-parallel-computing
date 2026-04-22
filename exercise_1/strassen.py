import numpy as np


def strassen(A, B):
    """
    Compute the product C = A x B using Strassen's recursive algorithm.

    Strassen reduces the number of recursive multiplications from 8 to 7
    by introducing additional additions and subtractions. This reduces the
    asymptotic complexity from O(n^3) to approximately O(n^2.807).

    The input matrices are padded to the next power of 2 if necessary,
    and the result is trimmed back to the original dimensions.

    Note: Strassen introduces overhead from additional additions and memory
    allocations. In practice it is only faster than the classical algorithm
    for very large matrices (typically n > 512).

    Args:
        A (list[list[float]]): Left matrix of shape (m, n).
        B (list[list[float]]): Right matrix of shape (n, p).

    Returns:
        list[list[float]]): Result matrix C of shape (m, p).
    """
    m, n, p = len(A), len(A[0]), len(B[0])

    # pad matrices to the next power of 2 for clean recursive splitting
    size = 1
    while size < max(m, n, p):
        size *= 2

    A_pad = [[0.0] * size for _ in range(size)]
    B_pad = [[0.0] * size for _ in range(size)]

    for i in range(m):
        for j in range(n):
            A_pad[i][j] = A[i][j]

    for i in range(n):
        for j in range(p):
            B_pad[i][j] = B[i][j]

    C_pad = _strassen_recursive(A_pad, B_pad)

    # trim result back to original dimensions
    return [C_pad[i][:p] for i in range(m)]


def _strassen_recursive(A, B):
    """
    Recursive core of Strassen's algorithm.

    Splits A and B into quadrants, computes the 7 Strassen products,
    and assembles the result. Base case uses a direct triple loop
    to avoid excessive recursion overhead on small matrices.

    Args:
        A (list[list[float]]): Square matrix of size n x n (n is a power of 2).
        B (list[list[float]]): Square matrix of size n x n (n is a power of 2).

    Returns:
        list[list[float]]): Result matrix of size n x n.
    """
    n = len(A)

    # base case: use classical multiplication for small matrices
    if n <= 64:
        C = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    mid = n // 2

    # split A and B into four quadrants each
    A11, A12, A21, A22 = _split(A, mid)
    B11, B12, B21, B22 = _split(B, mid)

    # compute the 7 Strassen products — each replaces one full matrix multiplication
    M1 = _strassen_recursive(_add(A11, A22), _add(B11, B22))
    M2 = _strassen_recursive(_add(A21, A22), B11)
    M3 = _strassen_recursive(A11, _sub(B12, B22))
    M4 = _strassen_recursive(A22, _sub(B21, B11))
    M5 = _strassen_recursive(_add(A11, A12), B22)
    M6 = _strassen_recursive(_sub(A21, A11), _add(B11, B12))
    M7 = _strassen_recursive(_sub(A12, A22), _add(B21, B22))

    # assemble the four quadrants of C from the 7 products
    C11 = _add(_sub(_add(M1, M4), M5), M7)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_sub(_add(M1, M3), M2), M6)

    return _join(C11, C12, C21, C22, mid)


def _split(M, mid):
    """Split matrix M into four quadrants at the given midpoint."""
    A11 = [row[:mid] for row in M[:mid]]
    A12 = [row[mid:] for row in M[:mid]]
    A21 = [row[:mid] for row in M[mid:]]
    A22 = [row[mid:] for row in M[mid:]]
    return A11, A12, A21, A22


def _join(C11, C12, C21, C22, mid):
    """Join four quadrants back into a single matrix."""
    size = mid * 2
    C = [[0.0] * size for _ in range(size)]
    for i in range(mid):
        for j in range(mid):
            C[i][j] = C11[i][j]
            C[i][j + mid] = C12[i][j]
            C[i + mid][j] = C21[i][j]
            C[i + mid][j + mid] = C22[i][j]
    return C


def _add(A, B):
    """Element-wise matrix addition."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _sub(A, B):
    """Element-wise matrix subtraction."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


if __name__ == "__main__":
    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]

    B = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]

    C = strassen(A, B)
    C_np = np.matmul(A, B)

    if np.allclose(C, C_np):
        print("Test 1 passed: Strassen multiplication is correct")
    else:
        print("Test 1 failed")