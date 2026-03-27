import numpy as np


def gf2_gauss_jordan(A, b):
    A = np.array(A, dtype=int, copy=True) % 2
    b = np.array(b, dtype=int, copy=True).reshape(-1, 1) % 2
    rows, cols = A.shape
    A = np.hstack((A, b))

    rank = 0
    free_vars = set(range(cols))
    for i in range(min(rows, cols)):
        if A[i, i] == 0:
            for j in range(i + 1, rows):
                if A[j, i] == 1:
                    A[[i, j]] = A[[j, i]]
                    break

        if A[i, i] == 0:
            continue

        free_vars.discard(i)

        for j in range(rows):
            if j != i and A[j, i] == 1:
                A[j] = A[j] ^ A[i]

        rank += 1

    x = A[:, -1]

    for row in A:
        if np.all(row[:-1] == 0) and row[-1] == 1:
            return None, "无解"

    if len(free_vars) > 0:
        return x, "无穷多解，存在自由变量"

    return x, "唯一解"
