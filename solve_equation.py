import numpy as np


def gf2_gauss_jordan(A, b):
    # 矩阵大小

    rows, cols = A.shape

    # 将 b 合并到 A 中，形成增广矩阵

    A = np.hstack((A, b.reshape(-1, 1)))

    rank = 0  # 矩阵的秩
    free_vars = set(range(cols))  # 自由变量的集合
    for i in range(min(rows, cols)):
        # 寻找主元，并将其所在的行交换到第 i 行
        if A[i, i] == 0:
            for j in range(i + 1, rows):
                if A[j, i] == 1:
                    A[[i, j]] = A[[j, i]]
                    break

        # 如果没有找到主元，继续下一列
        if A[i, i] == 0:
            continue

        # 标记当前列为非自由变量
        free_vars.discard(i)

        # 对其他行进行消元
        for j in range(rows):
            if j != i and A[j, i] == 1:
                A[j] = A[j] ^ A[i]  # 在 GF(2) 中加法是异或运算

        rank += 1  # 增加秩

    # 提取解向量
    x = A[:, -1]

    # 检查是否有无解的情况
    for row in A:
        if np.all(row[:-1] == 0) and row[-1] == 1:
            return None, '无解'

    # 检查是否有无穷多解
    if len(free_vars) > 0:
        return x, '无穷多解，存在自由变量'

    return x, '唯一解'



