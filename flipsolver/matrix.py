
import numpy as np


def generate_matrix(n):
    size = n * n
    A = np.zeros((size, size), dtype=int)
    for i in range(n):
        for j in range(n):
            current_index = i * n + j
            for col in range(n):
                A[current_index, i * n + col] = 1
            for row in range(n):
                A[current_index, row * n + j] = 1

    return A


def generate_matrix_2(n, matrix):
    A = generate_matrix(n)
    B = matrix
    C = [1] * (n * n)
    delete_indices = []
    for i in range(n):
        for j in range(n):
            k = i * n + j
            if B[i][j] == 2:
                delete_indices.append(k)
                C[k] = 2
            elif B[i][j] == 0:
                C[k] = 0

    delete_indices = sorted(delete_indices, reverse=True)
    for idx in delete_indices:
        A = np.delete(A, idx, axis=0)
        A = np.delete(A, idx, axis=1)

    dict_index = {}
    ans = 0
    for i in range(0, n * n):
        if C[i] != 2:
            dict_index[ans] = i
            ans += 1

    C = [x for x in C if x != 2]
    matrixC = np.array(C).reshape(-1, 1)

    return A, matrixC, len(C), dict_index
