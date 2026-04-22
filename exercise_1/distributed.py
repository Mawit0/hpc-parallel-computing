import numpy as np
from mpi4py import MPI


def distributed_multiply(A, B):
    """
    Compute the product C = A x B using distributed memory with mpi4py.

    Matrix A is split into row blocks and distributed across all available
    processes using MPI collective communication. Each process computes its
    partial result using the full matrix B, which is broadcast to all processes.
    Partial results are gathered back to the root process to form the final C.

    Data distribution:
        - Process 0 (root) holds the full matrices A and B initially.
        - B is broadcast to all processes using MPI Bcast.
        - A is split into row blocks and scattered using MPI Scatter.
        - Each process computes its local block of C.
        - Results are gathered back to root using MPI Gather.

    Args:
        A (list[list[float]]): Left matrix of shape (m, n). Only required on root.
        B (list[list[float]]): Right matrix of shape (n, p). Only required on root.

    Returns:
        list[list[float]] or None: Result matrix C of shape (m, p) on root process.
            Returns None on non-root processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()      # id of the current process
    size = comm.Get_size()      # total number of processes

    # broadcast B to all processes so every worker has the full matrix
    B = comm.bcast(B, root=0)

    # split A into row blocks and send one block to each process
    if rank == 0:
        chunk_size = len(A) // size
        A_chunks = [A[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
    else:
        A_chunks = None

    A_block = comm.scatter(A_chunks, root=0)

    # each process computes its local block of C using pure loops
    C_block = []
    for i in range(len(A_block)):
        row = []
        for j in range(len(B[0])):
            c_ij = 0
            for k in range(len(A_block[0])):
                c_ij += A_block[i][k] * B[k][j]  # accumulate dot product for position (i, j)
            row.append(c_ij)
        C_block.append(row)

    # gather all partial results back to root
    C_chunks = comm.gather(C_block, root=0)

    if rank == 0:
        C = []
        for block in C_chunks:
            C.extend(block)
        return C

    return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9],
         [10, 11, 12]] if rank == 0 else None

    B = [[1, 2],
         [3, 4],
         [5, 6]] if rank == 0 else None

    C = distributed_multiply(A, B)

    if rank == 0:
        C_np = np.matmul(A, B)
        if np.allclose(C, C_np):
            print("Test 1 passed: distributed matrix multiplication is correct")
        else:
            print("Test 1 failed")