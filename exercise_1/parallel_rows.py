import numpy as np
from multiprocessing import Pool


def multiply_rows_worker(args):
    """
    Compute the partial matrix product for a block of rows.

    This function is executed by a single worker in the Pool. It receives
    a block of rows from A and the full matrix B, and returns the
    corresponding rows of the result matrix C.

    Args:
        args (tuple): A tuple containing:
            - A_block (list[list[float]]): A subset of rows from matrix A.
            - B (list[list[float]]): The full matrix B of shape (n, p).

    Returns:
        list[list[float]]: Partial result matrix of shape (len(A_block), p).
    """
    A_block, B = args

    C = []
    for i in range(len(A_block)):
        row = []
        for j in range(len(B[0])):
            c_ij = 0
            for k in range(len(A_block[0])):
                c_ij += A_block[i][k] * B[k][j]  # accumulate dot product for position (i, j)
            row.append(c_ij)
        C.append(row)

    return C


def parallel_multiply_rows(A, B, num_workers):
    """
    Compute the product C = A x B using row-based partitioning with multiprocessing.

    Matrix A is split into equally sized row blocks, one per worker. Each worker
    computes its partial result independently using the full matrix B. Results
    are collected and concatenated in order to form the final matrix C.

    Args:
        A (list[list[float]]): Left matrix of shape (m, n).
        B (list[list[float]]): Right matrix of shape (n, p).
        num_workers (int): Number of parallel worker processes.

    Returns:
        list[list[float]]: Result matrix C of shape (m, p).
    """
    chunk_size = len(A) // num_workers
    # pair each row block with the full B so each worker receives a single argument
    chunks = [(A[i * chunk_size:(i + 1) * chunk_size], B) for i in range(num_workers)]

    with Pool(processes=num_workers) as pool:
        results = pool.map(multiply_rows_worker, chunks)

    C = []
    for block in results:
        C.extend(block)

    return C


if __name__ == "__main__":
    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9],
         [10, 11, 12]]

    B = [[1, 2],
         [3, 4],
         [5, 6]]

    c = parallel_multiply_rows(A, B, num_workers=2)
    c_np = np.matmul(A, B)

    if np.allclose(c, c_np):
        print("Test 1 passed: row-based parallel multiplication is correct")
    else:
        print("Test 1 failed")