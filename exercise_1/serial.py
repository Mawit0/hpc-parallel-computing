import numpy as np

def matrix_multiply(A, B):
    """
    Compute the product C = A x B using nested loops.

    This is the serial baseline implementation. It follows the definition
    c_ij = sum_k(a_ik * b_kj) without any numerical optimizations,
    serving as the reference for correctness and performance comparisons.

    Args:
        A (list[list[float]]): Left matrix of shape (m, n).
        B (list[list[float]]): Right matrix of shape (n, p).

    Returns:
        list[list[float]]: Result matrix C of shape (m, p).

    Raises:
        ValueError: If the number of columns in A does not match the number of rows in rows in B.
    """
    if len(A[0]) != len(B):
        raise ValueError(
            f"Incompatible dimensions: A has {len(A[0])} columns but B has {len(B)} rows."
        )

    C = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            c_ij = 0
            for k in range(len(A[0])):
                c_ij += A[i][k] * B[k][j]  # accumulate dot product for position (i, j)
            row.append(c_ij)
        C.append(row)

    return C


if __name__ == "__main__":
    # Test 1: known small matrices where result can be verified by hand
    A = [[1, 2, 3],
         [4, 5, 6]]

    B = [[1, 2],
         [3, 4],
         [5, 6]]

    c = matrix_multiply(A, B)
    c_np = np.matmul(A, B)

    if np.allclose(c, c_np):
        print("Test 1 passed: 2x3 @ 3x2 multiplication is correct")
    else:
        print("Test 1 failed")

    # Test 2: incompatible dimensions must raise ValueError
    try:
        matrix_multiply([[1, 2], [3, 4]], [[1, 2], [3, 4], [5, 6]])
        print("Test 2 failed: ValueError was not raised")
    except ValueError as e:
        print(f"Test 2 passed: {e}")