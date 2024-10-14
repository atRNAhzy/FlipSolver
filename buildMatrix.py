

import numpy as np

# 修改后的生成矩阵函数，标记当前点和同行与同列的所有点
def generate_matrix(n):
    size = n * n  # 计算输出矩阵的大小
    A = np.zeros((size, size), dtype=int)  # 初始化矩阵A
    for i in range(n):
        for j in range(n):
            current_index = i * n + j
                # 标记当前点及其所在行和列的所有点
                # 标记当前行
            for col in range(n):
                A[current_index, i * n + col] = 1  # 同行的点
                # 标记当前列
            for row in range(n):
                A[current_index, row * n + j] = 1  # 同列的点

    return A

def generate_matrix_2(n,matrix):
    A=generate_matrix(n)
    B=matrix
    C = [1] * (n * n)
    delete_indices = []  # 用于存储需要删除的行列索引
    # 遍历矩阵 A
    for i in range(n):
        for j in range(n):
            k = i * n + j  # 当前是第 k 个元素
            if B[i][j] == 2:
                delete_indices.append(k)  # 标记需要删除的行列
                C[k] = 2  # 更新 C[k] 为 0
            elif B[i][j] == 0:
                C[k] = 0

    # 删除 A 中对应的行和列
    delete_indices = sorted(delete_indices, reverse=True)  # 逆序删除，避免索引混乱
    for idx in delete_indices:
        A = np.delete(A, idx, axis=0)  # 删除第 idx 行
        A = np.delete(A, idx, axis=1)  # 删除第 idx
    #保存索引用于坐标转换
    dict_index={}
    ans=0
    for i in range(0,n*n):
        if C[i] !=2:
            dict_index[ans]=i
            ans+=1

    # 删除列表中的多余元素并转换为n*1矩阵
    C = [x for x in C if x != 2]
    matrixC = np.array(C).reshape(-1, 1)


    return A, matrixC, len(C),dict_index


