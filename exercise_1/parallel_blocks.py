import numpy as np
from multiprocessing import Pool


def multiply_blocks_worker(args):
    """
    Compute the partial matrix product for a block defined by a row and column range.

    This function is executed by a single worker in the Pool. It receives
    a block of rows from A and a block of columns from B, and returns the
    corresponding submatrix of C.

    Args:
        args (tuple): A tuple containing:
            - A_block (list[list[float]]): A subset of rows from matrix A.
            - B_block (list[list[float]]): A subset of columns from matrix B.
            - row_start (int): Starting row index in the full matrix C.
            - col_start (int): Starting column index in the full matrix C.

    Returns:
        tuple: A tuple containing:
            - row_start (int): Starting row index in C.
            - col_start (int): Starting column index in C.
            - block (list[list[float]]): Computed submatrix of C.
    """
    A_block, B_block, row_start, col_start = args

    block = []
    for i in range(len(A_block)):
        row = []
        for j in range(len(B_block[0])):
            c_ij = 0
            for k in range(len(A_block[0])):
                c_ij += A_block[i][k] * B_block[k][j]  # accumulate dot product for position (i, j)
            row.append(c_ij)
        block.append(row)

    return row_start, col_start, block


def parallel_multiply_blocks(A, B, num_workers):
    """
    Compute the product C = A x B using 2D block partitioning with multiprocessing.

    Both A and B are divided into blocks along rows and columns respectively.
    Each worker computes one block of the result matrix C. Results are placed
    back into C using the row and column offsets returned by each worker.

    A 2D decomposition was chosen over 1D because it distributes both row and
    column work evenly, reducing the size of data each worker handles and
    improving cache locality.

    Args:
        A (list[list[float]]): Left matrix of shape (m, n).
        B (list[list[float]]): Right matrix of shape (n, p).
        num_workers (int): Number of parallel worker processes. Must be a perfect square.

    Returns:
        list[list[float]]): Result matrix C of shape (m, p).
    """
    import math
    grid = int(math.sqrt(num_workers))  # number of blocks per dimension

    row_chunk = len(A) // grid
    col_chunk = len(B[0]) // grid

    # transpose B to extract column blocks easily
    B_T = list(map(list, zip(*B)))

    chunks = []
    for i in range(grid):
        A_block = A[i * row_chunk:(i + 1) * row_chunk]
        for j in range(grid):
            B_block = list(map(list, zip(*B_T[j * col_chunk:(j + 1) * col_chunk])))
            chunks.append((A_block, B_block, i * row_chunk, j * col_chunk))

    with Pool(processes=num_workers) as pool:
        results = pool.map(multiply_blocks_worker, chunks)

    # initialize empty C and fill in each block using its offset
    m = len(A)
    p = len(B[0])
    C = [[0.0] * p for _ in range(m)]

    for row_start, col_start, block in results:
        for i, row in enumerate(block):
            for j, val in enumerate(row):
                C[row_start + i][col_start + j] = val

    return C


if __name__ == "__main__":
    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]

    B = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]

    c = parallel_multiply_blocks(A, B, num_workers=4)
    c_np = np.matmul(A, B)

    if np.allclose(c, c_np):
        print("Test 1 passed: block-based parallel multiplication is correct")
    else:
        print("Test 1 failed")