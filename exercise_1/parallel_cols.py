import numpy as np
from multiprocessing import Pool


def multiply_cols_worker(args):
    """
    Compute the partial matrix product for a block of columns.

    This function is executed by a single worker in the Pool. It receives
    the full matrix A and a block of columns from B, and returns the
    corresponding columns of the result matrix C.

    Args:
        args (tuple): A tuple containing:
            - A (list[list[float]]): The full matrix A of shape (m, n).
            - B_block (list[list[float]]): A subset of columns from matrix B.

    Returns:
        list[list[float]]: Partial result matrix of shape (m, len(B_block[0])).
    """
    A, B_block = args

    C = []
    for i in range(len(A)):
        row = []
        for j in range(len(B_block[0])):
            c_ij = 0
            for k in range(len(A[0])):
                c_ij += A[i][k] * B_block[k][j]  # accumulate dot product for position (i, j)
            row.append(c_ij)
        C.append(row)

    return C


def parallel_multiply_cols(A, B, num_workers):
    """
    Compute the product C = A x B using column-based partitioning with multiprocessing.

    Matrix B is split into equally sized column blocks, one per worker. Each worker
    computes its partial result independently using the full matrix A. Results
    are collected and concatenated column-wise to form the final matrix C.

    Args:
        A (list[list[float]]): Left matrix of shape (m, n).
        B (list[list[float]]): Right matrix of shape (n, p).
        num_workers (int): Number of parallel worker processes.

    Returns:
        list[list[float]]: Result matrix C of shape (m, p).
    """
    num_cols = len(B[0])
    chunk_size = num_cols // num_workers

    # split B into column blocks by transposing, slicing rows, then transposing back
    B_T = list(map(list, zip(*B)))
    chunks = [(A, list(map(list, zip(*B_T[i * chunk_size:(i + 1) * chunk_size])))) 
              for i in range(num_workers)]

    with Pool(processes=num_workers) as pool:
        results = pool.map(multiply_cols_worker, chunks)

    # concatenate results column-wise
    C = []
    for i in range(len(A)):
        row = []
        for block in results:
            row.extend(block[i])
        C.append(row)

    return C


if __name__ == "__main__":
    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9],
         [10, 11, 12]]

    B = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]]

    c = parallel_multiply_cols(A, B, num_workers=2)
    c_np = np.matmul(A, B)

    if np.allclose(c, c_np):
        print("Test 1 passed: column-based parallel multiplication is correct")
    else:
        print("Test 1 failed")