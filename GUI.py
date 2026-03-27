import sys

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from buildMatrix import generate_matrix, generate_matrix_2
from solve_equation import gf2_gauss_jordan


APP_STYLE = """
QWidget {
    background: #f4efe7;
    color: #26211b;
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
}
QFrame#panel {
    background: #fbf8f3;
    border: 1px solid #e5ddd1;
    border-radius: 24px;
}
QFrame#boardPanel {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #fffaf2,
        stop: 1 #f2eadf
    );
    border: 1px solid #e0d5c5;
    border-radius: 28px;
}
QLineEdit {
    background: #fffdf8;
    border: 1px solid #d7ccbb;
    border-radius: 14px;
    padding: 12px 14px;
    font-size: 14px;
}
QTextEdit {
    background: #fffdf8;
    border: 1px solid #ded3c4;
    border-radius: 18px;
    padding: 10px;
    font-size: 13px;
}
QPushButton {
    border: none;
    border-radius: 16px;
    padding: 12px 16px;
    background: #e8dece;
    color: #2d261f;
    font-weight: 600;
}
QPushButton:hover {
    background: #ddd0bb;
}
QPushButton:pressed {
    background: #d0c0a6;
}
QPushButton#primaryButton {
    background: #1f6f5f;
    color: white;
}
QPushButton#primaryButton:hover {
    background: #185c4f;
}
QPushButton#accentButton {
    background: #c96d42;
    color: white;
}
QPushButton#accentButton:hover {
    background: #ae5d37;
}
QPushButton#ghostButton {
    background: transparent;
    border: 1px solid #d5cab8;
}
QRadioButton {
    spacing: 8px;
    font-size: 13px;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
}
QRadioButton::indicator:unchecked {
    border: 2px solid #c5b8a4;
    border-radius: 8px;
    background: #fffaf2;
}
QRadioButton::indicator:checked {
    border: 2px solid #1f6f5f;
    border-radius: 8px;
    background: #1f6f5f;
}
QLabel#titleLabel {
    font-size: 30px;
    font-weight: 800;
    color: #201a14;
}
QLabel#subtitleLabel {
    font-size: 14px;
    color: #6a5e52;
}
QLabel#sectionTitle {
    font-size: 16px;
    font-weight: 700;
}
QLabel#hintLabel {
    font-size: 12px;
    color: #7a6f62;
}
QLabel#statusBadge {
    background: #efe5d5;
    color: #74562f;
    border-radius: 14px;
    padding: 8px 12px;
    font-weight: 700;
}
QPushButton[cellState="0"] {
    background: #fffef9;
    border: 1px solid #dacdbb;
}
QPushButton[cellState="1"] {
    background: #2b211b;
    border: 1px solid #2b211b;
}
QPushButton[cellState="2"] {
    background: #cdbfab;
    border: 1px dashed #af9f89;
    color: #6f6356;
}
"""


def output_coordinates(matrix, n):
    grid = matrix.reshape((n, n))
    coordinates = []
    for i in range(n):
        for j in range(n):
            if grid[i, j] == 1:
                coordinates.append((i + 1, j + 1))
    return coordinates


def find_coordinates(index, n):
    return index // n + 1, index % n + 1


class Worker(QThread):
    processed = pyqtSignal(str)

    def __init__(self, matrix, n, matrix0, mode):
        super().__init__()
        self.matrix = matrix
        self.n_value = n
        self.matrix0 = matrix0
        self.mode = mode

    def run(self):
        if self.mode == 1:
            A = generate_matrix(self.n_value)
            x, status = gf2_gauss_jordan(A, self.matrix)
            if status != "无解":
                coords = output_coordinates(x, self.n_value)
                lines = [status, "", "建议点击坐标"]
                if coords:
                    lines.extend([f"{idx + 1}. ({row}, {col})" for idx, (row, col) in enumerate(coords)])
                else:
                    lines.append("当前棋盘已经是目标状态，无需操作。")
            else:
                lines = [status, "", "当前状态下不存在可行解。"]
        else:
            A_1, matrix, _, dict_index = generate_matrix_2(self.n_value, self.matrix0)
            x, status = gf2_gauss_jordan(A_1, matrix)
            if status != "无解":
                lines = [status, "", "建议点击坐标"]
                for index, value in enumerate(x):
                    if value == 1:
                        row, col = find_coordinates(dict_index[index], self.n_value)
                        lines.append(f"{len(lines) - 2}. ({row}, {col})")
                if len(lines) == 3:
                    lines.append("当前棋盘已经是目标状态，无需操作。")
            else:
                lines = [status, "", "异形模式下该局面不存在可行解。"]

        self.processed.emit("\n".join(lines))


class GridApp(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = None
        self.grid_state = []
        self.cell_buttons = []
        self.started = False
        self.n_value = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("FlipSolver")
        self.resize(1180, 760)
        self.setStyleSheet(APP_STYLE)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(28, 24, 28, 24)
        root_layout.setSpacing(20)

        hero_panel = QFrame()
        hero_panel.setObjectName("panel")
        hero_layout = QVBoxLayout(hero_panel)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(8)

        title_label = QLabel("FlipSolver")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel(
            "求解行列翻转棋盘。先设置局面，再切换到求解阶段，让线性方程组给出可行点击序列。"
        )
        subtitle_label.setWordWrap(True)
        subtitle_label.setObjectName("subtitleLabel")

        hero_layout.addWidget(title_label)
        hero_layout.addWidget(subtitle_label)
        root_layout.addWidget(hero_panel)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        root_layout.addLayout(content_layout)

        control_panel = QFrame()
        control_panel.setObjectName("panel")
        control_panel.setFixedWidth(340)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(22, 22, 22, 22)
        control_layout.setSpacing(16)

        grid_title = QLabel("棋盘设置")
        grid_title.setObjectName("sectionTitle")
        control_layout.addWidget(grid_title)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("输入网格大小，例如 5")
        self.input_box.returnPressed.connect(self.create_grid)
        control_layout.addWidget(self.input_box)

        size_hint = QLabel("建议使用 3 到 8 的大小，便于观察和求解。")
        size_hint.setObjectName("hintLabel")
        size_hint.setWordWrap(True)
        control_layout.addWidget(size_hint)

        self.generate_button = QPushButton("生成棋盘", self)
        self.generate_button.setObjectName("primaryButton")
        self.generate_button.clicked.connect(self.create_grid)
        control_layout.addWidget(self.generate_button)

        mode_title = QLabel("求解模式")
        mode_title.setObjectName("sectionTitle")
        control_layout.addWidget(mode_title)

        self.mode_group = QButtonGroup(self)
        self.classic_mode = QRadioButton("经典模式")
        self.classic_mode.setChecked(True)
        self.special_mode = QRadioButton("异形模式")
        self.mode_group.addButton(self.classic_mode, 1)
        self.mode_group.addButton(self.special_mode, 2)
        control_layout.addWidget(self.classic_mode)
        control_layout.addWidget(self.special_mode)

        mode_hint = QLabel("异形模式会把非黑色格子锁定成不可操作区域。")
        mode_hint.setObjectName("hintLabel")
        mode_hint.setWordWrap(True)
        control_layout.addWidget(mode_hint)

        action_title = QLabel("操作流程")
        action_title.setObjectName("sectionTitle")
        control_layout.addWidget(action_title)

        self.start_button = QPushButton("开始求解阶段", self)
        self.start_button.clicked.connect(self.start_game)
        self.solve_button = QPushButton("计算解", self)
        self.solve_button.setObjectName("accentButton")
        self.solve_button.clicked.connect(self.save_state)
        self.end_button = QPushButton("回到编辑阶段", self)
        self.end_button.clicked.connect(self.end_game)
        self.reset_button = QPushButton("清空棋盘", self)
        self.reset_button.setObjectName("ghostButton")
        self.reset_button.clicked.connect(self.reset_board)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.solve_button)
        control_layout.addWidget(self.end_button)
        control_layout.addWidget(self.reset_button)

        self.status_badge = QLabel("状态: 编辑中")
        self.status_badge.setObjectName("statusBadge")
        control_layout.addWidget(self.status_badge)

        result_title = QLabel("求解结果")
        result_title.setObjectName("sectionTitle")
        control_layout.addWidget(result_title)

        self.result_box = QTextEdit(self)
        self.result_box.setReadOnly(True)
        self.result_box.setPlaceholderText("点击“计算解”后，这里会显示一组建议坐标。")
        self.result_box.setMinimumHeight(220)
        control_layout.addWidget(self.result_box)

        self.copy_button = QPushButton("复制结果", self)
        self.copy_button.setObjectName("ghostButton")
        self.copy_button.clicked.connect(self.copy_result)
        control_layout.addWidget(self.copy_button)
        control_layout.addStretch(1)

        content_layout.addWidget(control_panel)

        board_panel = QFrame()
        board_panel.setObjectName("boardPanel")
        board_layout = QVBoxLayout(board_panel)
        board_layout.setContentsMargins(24, 24, 24, 24)
        board_layout.setSpacing(14)

        board_header = QHBoxLayout()
        board_header.setSpacing(12)

        board_title_box = QVBoxLayout()
        board_title_box.setSpacing(4)
        board_title = QLabel("棋盘")
        board_title.setObjectName("sectionTitle")
        board_subtitle = QLabel("编辑阶段单点切换，求解阶段按规则翻转同行、同列与自身。")
        board_subtitle.setObjectName("hintLabel")
        board_subtitle.setWordWrap(True)
        board_title_box.addWidget(board_title)
        board_title_box.addWidget(board_subtitle)

        self.board_meta = QLabel("尚未生成棋盘")
        self.board_meta.setObjectName("statusBadge")
        self.board_meta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        board_header.addLayout(board_title_box, 1)
        board_header.addWidget(self.board_meta)
        board_layout.addLayout(board_header)

        self.board_hint = QLabel("先输入大小并生成棋盘。")
        self.board_hint.setObjectName("hintLabel")
        board_layout.addWidget(self.board_hint)

        self.board_container = QWidget()
        self.board_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.board_wrapper = QVBoxLayout(self.board_container)
        self.board_wrapper.setContentsMargins(0, 10, 0, 0)
        self.board_wrapper.setAlignment(Qt.AlignCenter)
        board_layout.addWidget(self.board_container, 1)

        content_layout.addWidget(board_panel, 1)

    def create_grid(self):
        if self.grid is not None:
            self.clear_existing_grid()

        try:
            n = int(self.input_box.text())
            if n <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的正整数网格大小。")
            return

        self.n_value = n
        self.started = False
        self.grid_state = [0] * (n * n)
        self.cell_buttons = []
        self.result_box.clear()

        self.grid = QGridLayout()
        self.grid.setSpacing(8)
        self.grid.setAlignment(Qt.AlignCenter)

        button_size = max(34, min(62, 310 // max(1, n) + 16))

        for i in range(n):
            for j in range(n):
                index = i * n + j
                button = QPushButton()
                button.setCursor(Qt.PointingHandCursor)
                button.setFixedSize(button_size, button_size)
                button.clicked.connect(lambda _, idx=index: self.toggle_state(idx))
                self.cell_buttons.append(button)
                self.set_cell_state(index, 0)
                self.grid.addWidget(button, i, j)

        board_widget = QWidget()
        board_widget.setLayout(self.grid)
        self.board_wrapper.addWidget(board_widget, alignment=Qt.AlignCenter)

        self.update_status_text("编辑中")
        self.board_meta.setText(f"{n} x {n}")
        self.board_hint.setText("白色表示关闭，深色表示开启，灰色表示异形模式下的锁定区域。")

    def clear_existing_grid(self):
        while self.board_wrapper.count():
            item = self.board_wrapper.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.grid = None

    def set_cell_state(self, index, state):
        self.grid_state[index] = state
        button = self.cell_buttons[index]
        button.setProperty("cellState", str(state))
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()
        button.setEnabled(state != 2)

    def toggle_state(self, index):
        if not self.grid_state:
            return

        if self.started:
            self.apply_move(index)
        else:
            current = self.grid_state[index]
            if current == 2:
                return
            self.set_cell_state(index, 0 if current == 1 else 1)

    def apply_move(self, index):
        n = self.n_value
        row, col = divmod(index, n)
        affected = set()

        for i in range(n):
            affected.add(row * n + i)
            affected.add(i * n + col)
        affected.add(index)

        for affected_index in affected:
            if self.grid_state[affected_index] == 2:
                continue
            self.set_cell_state(affected_index, 0 if self.grid_state[affected_index] == 1 else 1)

    def start_game(self):
        if not self.grid_state:
            QMessageBox.warning(self, "提示", "请先生成棋盘。")
            return

        self.started = True
        if self.special_mode.isChecked():
            self.lock_non_black_buttons()
            self.board_hint.setText("异形模式已生效：只有深色格子保留为可操作区域。")
        else:
            self.board_hint.setText("求解阶段中，点击任意格子会翻转同行、同列与自身。")
        self.update_status_text("求解阶段")

    def end_game(self):
        if not self.grid_state:
            return
        self.started = False
        self.update_status_text("编辑中")
        self.board_hint.setText("已回到编辑阶段，可以继续调整初始局面。")

    def lock_non_black_buttons(self):
        for index, value in enumerate(self.grid_state):
            if value != 1:
                self.set_cell_state(index, 2)

    def reset_board(self):
        if not self.grid_state:
            return

        self.started = False
        for index in range(len(self.grid_state)):
            self.set_cell_state(index, 0)
        self.result_box.clear()
        self.update_status_text("编辑中")
        self.board_hint.setText("棋盘已清空，可以重新设置局面。")

    def save_state(self):
        if not self.grid_state:
            QMessageBox.warning(self, "提示", "请先生成棋盘。")
            return

        if self.special_mode.isChecked():
            self.lock_non_black_buttons()

        mode = self.mode_group.checkedId()
        matrix = np.array(self.grid_state).reshape(-1, 1)
        matrix0 = matrix.reshape((self.n_value, self.n_value))
        self.run_worker(matrix, self.n_value, matrix0, mode)

    def run_worker(self, matrix, n, matrix0, mode):
        self.solve_button.setEnabled(False)
        self.result_box.setPlainText("正在计算，请稍候...")
        self.worker = Worker(matrix, n, matrix0, mode)
        self.worker.processed.connect(self.handle_processed)
        self.worker.finished.connect(lambda: self.solve_button.setEnabled(True))
        self.worker.start()

    def handle_processed(self, result):
        self.result_box.setPlainText(result)
        lines = [line.strip() for line in result.splitlines() if line.strip()]
        if lines:
            self.board_hint.setText(f"求解完成: {lines[0]}")

    def update_status_text(self, text):
        self.status_badge.setText(f"状态: {text}")

    def copy_result(self):
        text = self.result_box.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "当前没有可复制的结果。")
            return
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "已复制", "求解结果已复制到剪贴板。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GridApp()
    window.show()
    sys.exit(app.exec_())
