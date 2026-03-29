import os
import unittest

import numpy as np

from flipsolver.levels import build_random_classic_level
from flipsolver.matrix import generate_matrix

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtWidgets import QApplication
    from flipsolver.gui import GridApp, STATE_BLACK, STATE_BLOCKED, STATE_WHITE
    GUI_AVAILABLE = True
except Exception:  # pragma: no cover - runtime dependency check
    GUI_AVAILABLE = False


class TestRandomClassicLevel(unittest.TestCase):
    def test_random_classic_level_is_solvable_by_construction(self):
        rng = np.random.default_rng(123)
        n = 4
        level = np.array(build_random_classic_level(n, rng), dtype=int).reshape(-1, 1)
        A = generate_matrix(n) % 2

        # 随机关卡来自 A @ moves，因此至少存在一个解。
        # 这里通过穷举验证：存在点击向量 x 使得 A @ x == level (mod 2)。
        found = False
        for value in range(1 << (n * n)):
            x = np.array([(value >> bit) & 1 for bit in range(n * n)], dtype=int).reshape(-1, 1)
            if np.array_equal((A @ x) % 2, level % 2):
                found = True
                break

        self.assertTrue(found)


@unittest.skipUnless(GUI_AVAILABLE, "PyQt5 GUI runtime is unavailable")
class TestGridAppResetBoard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_reset_board_keeps_blocked_cells_in_irregular_mode(self):
        window = GridApp()
        window.special_mode.setChecked(True)
        state = [STATE_BLACK, STATE_BLOCKED, STATE_WHITE, STATE_BLACK]
        window.create_grid_with_state(2, state)

        window.reset_board()

        self.assertEqual(window.grid_state, [STATE_WHITE, STATE_BLOCKED, STATE_WHITE, STATE_WHITE])


if __name__ == "__main__":
    unittest.main()
