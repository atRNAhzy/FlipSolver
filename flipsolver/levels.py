import numpy as np

from .matrix import generate_matrix, generate_matrix_2


STATE_BLACK = 1
STATE_BLOCKED = 2


def build_random_classic_level(n, rng):
    A = generate_matrix(n) % 2
    moves = rng.integers(0, 2, size=(n * n, 1), dtype=int)
    if not moves.any():
        moves[rng.integers(0, n * n), 0] = 1
    board = (A @ moves) % 2
    return board.reshape(-1).tolist()


def build_random_irregular_level(n, rng):
    min_active = max(3, n)
    for _ in range(100):
        active_mask = rng.random(n * n) > 0.35
        if active_mask.sum() < min_active:
            continue

        template = np.full(n * n, STATE_BLOCKED, dtype=int)
        template[active_mask] = STATE_BLACK
        reduced_A, _, _, dict_index = generate_matrix_2(n, template.reshape((n, n)))

        moves = rng.integers(0, 2, size=(active_mask.sum(), 1), dtype=int)
        if not moves.any():
            moves[rng.integers(0, active_mask.sum()), 0] = 1

        board = np.full(n * n, STATE_BLOCKED, dtype=int)
        board_values = (reduced_A @ moves) % 2
        for reduced_index, original_index in dict_index.items():
            board[original_index] = int(board_values[reduced_index, 0])

        return board.tolist()

    board = np.full(n * n, STATE_BLOCKED, dtype=int)
    keep = rng.choice(n * n, size=min_active, replace=False)
    board[keep] = 1
    return board.tolist()
