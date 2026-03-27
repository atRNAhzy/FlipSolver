import sys
from string import Template

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QBoxLayout,
    QButtonGroup,
    QComboBox,
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

from .levels import build_random_classic_level, build_random_irregular_level
from .matrix import generate_matrix, generate_matrix_2
from .solver import gf2_gauss_jordan


STYLE_SCALES = {
    "标准": 1.0,
    "大字体": 1.18,
    "演示模式": 1.36,
}


def make_app_style(scale):
    return Template(
        """
QWidget {
    background: #f4efe7;
    color: #26211b;
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
    font-size: ${base_font}px;
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
    padding: ${input_padding_y}px ${input_padding_x}px;
    font-size: ${input_font}px;
}
QTextEdit {
    background: #fffdf8;
    border: 1px solid #ded3c4;
    border-radius: 18px;
    padding: ${text_padding}px;
    font-size: ${text_font}px;
}
QPushButton {
    border: none;
    border-radius: 16px;
    padding: ${button_padding_y}px ${button_padding_x}px;
    background: #e8dece;
    color: #2d261f;
    font-weight: 600;
    font-size: ${button_font}px;
    min-height: ${button_height}px;
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
    font-size: ${radio_font}px;
}
QRadioButton::indicator {
    width: ${radio_indicator}px;
    height: ${radio_indicator}px;
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
    font-size: ${title_font}px;
    font-weight: 800;
    color: #201a14;
}
QLabel#subtitleLabel {
    font-size: ${subtitle_font}px;
    color: #6a5e52;
}
QLabel#sectionTitle {
    font-size: ${section_font}px;
    font-weight: 700;
}
QLabel#hintLabel {
    font-size: ${hint_font}px;
    color: #7a6f62;
}
QLabel#statusBadge {
    background: #efe5d5;
    color: #74562f;
    border-radius: 14px;
    padding: ${badge_padding_y}px ${badge_padding_x}px;
    font-weight: 700;
    font-size: ${badge_font}px;
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
    ).substitute(
        base_font=int(17 * scale),
        input_padding_y=int(14 * scale),
        input_padding_x=int(16 * scale),
        input_font=int(19 * scale),
        text_padding=int(14 * scale),
        text_font=int(17 * scale),
        button_padding_y=int(14 * scale),
        button_padding_x=int(18 * scale),
        button_font=int(17 * scale),
        button_height=int(54 * scale),
        radio_font=int(17 * scale),
        radio_indicator=int(20 * scale),
        title_font=int(42 * scale),
        subtitle_font=int(20 * scale),
        section_font=int(24 * scale),
        hint_font=int(16 * scale),
        badge_padding_y=int(10 * scale),
        badge_padding_x=int(14 * scale),
        badge_font=int(16 * scale),
    )


STATE_WHITE = 0
STATE_BLACK = 1
STATE_BLOCKED = 2


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
                solution_count = 0
                for index, value in enumerate(x):
                    if value == 1:
                        solution_count += 1
                        row, col = find_coordinates(dict_index[index], self.n_value)
                        lines.append(f"{solution_count}. ({row}, {col})")
                if solution_count == 0:
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
        self.rng = np.random.default_rng()
        self.current_scale = STYLE_SCALES["演示模式"]
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("FlipSolver")
        self.resize(1180, 760)
        self.apply_scale_style()

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
            "求解行列翻转棋盘。支持随机经典关卡和随机异形关卡，可以直接进入游玩与求解。"
        )
        subtitle_label.setWordWrap(True)
        subtitle_label.setObjectName("subtitleLabel")

        hero_layout.addWidget(title_label)
        hero_layout.addWidget(subtitle_label)
        root_layout.addWidget(hero_panel)

        self.content_layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.content_layout.setSpacing(20)
        root_layout.addLayout(self.content_layout)

        control_panel = QFrame()
        self.control_panel = control_panel
        control_panel.setObjectName("panel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(22, 22, 22, 22)
        control_layout.setSpacing(16)
        control_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        control_panel.setMinimumWidth(340)

        grid_title = QLabel("棋盘设置")
        grid_title.setObjectName("sectionTitle")
        control_layout.addWidget(grid_title)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("输入网格大小，例如 5")
        self.input_box.returnPressed.connect(self.create_grid)
        control_layout.addWidget(self.input_box)

        size_hint = QLabel("建议使用 4 到 8 的大小，既有可玩性也方便观察。")
        size_hint.setObjectName("hintLabel")
        size_hint.setWordWrap(True)
        control_layout.addWidget(size_hint)

        self.generate_button = QPushButton("生成空白棋盘", self)
        self.generate_button.setObjectName("primaryButton")
        self.generate_button.clicked.connect(self.create_grid)
        control_layout.addWidget(self.generate_button)

        mode_title = QLabel("模式")
        mode_title.setObjectName("sectionTitle")
        control_layout.addWidget(mode_title)

        self.mode_group = QButtonGroup(self)
        self.classic_mode = QRadioButton("经典模式")
        self.classic_mode.setChecked(True)
        self.special_mode = QRadioButton("异形模式")
        self.mode_group.addButton(self.classic_mode, 1)
        self.mode_group.addButton(self.special_mode, 2)
        self.classic_mode.toggled.connect(self.handle_mode_change)
        control_layout.addWidget(self.classic_mode)
        control_layout.addWidget(self.special_mode)

        mode_hint = QLabel("经典模式只有黑白两态；异形模式允许灰色禁用格。")
        mode_hint.setObjectName("hintLabel")
        mode_hint.setWordWrap(True)
        control_layout.addWidget(mode_hint)

        scale_title = QLabel("显示缩放")
        scale_title.setObjectName("sectionTitle")
        control_layout.addWidget(scale_title)

        self.scale_box = QComboBox(self)
        self.scale_box.addItems(STYLE_SCALES.keys())
        self.scale_box.setCurrentText("演示模式")
        self.scale_box.currentTextChanged.connect(self.change_scale)
        control_layout.addWidget(self.scale_box)

        random_title = QLabel("随机关卡")
        random_title.setObjectName("sectionTitle")
        control_layout.addWidget(random_title)

        self.random_current_button = QPushButton("随机生成当前模式关卡", self)
        self.random_current_button.setObjectName("accentButton")
        self.random_current_button.clicked.connect(self.randomize_current_mode)
        self.random_classic_button = QPushButton("随机经典关卡", self)
        self.random_classic_button.clicked.connect(self.randomize_classic_mode)
        self.random_special_button = QPushButton("随机异形关卡", self)
        self.random_special_button.clicked.connect(self.randomize_special_mode)
        control_layout.addWidget(self.random_current_button)
        control_layout.addWidget(self.random_classic_button)
        control_layout.addWidget(self.random_special_button)

        action_title = QLabel("操作流程")
        action_title.setObjectName("sectionTitle")
        control_layout.addWidget(action_title)

        self.start_button = QPushButton("开始游玩", self)
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

        self.content_layout.addWidget(control_panel, 0)

        board_panel = QFrame()
        self.board_panel = board_panel
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
        board_subtitle = QLabel("编辑阶段可手动布置关卡；游玩阶段点击格子会翻转同行、同列与自身。")
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

        self.board_hint = QLabel("先输入大小并生成棋盘，或者直接随机一局。")
        self.board_hint.setObjectName("hintLabel")
        self.board_hint.setWordWrap(True)
        board_layout.addWidget(self.board_hint)

        self.board_container = QWidget()
        self.board_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.board_wrapper = QVBoxLayout(self.board_container)
        self.board_wrapper.setContentsMargins(0, 10, 0, 0)
        self.board_wrapper.setAlignment(Qt.AlignCenter)
        board_layout.addWidget(self.board_container, 1)

        self.content_layout.addWidget(board_panel, 1)
        self.update_layout_mode()

    def current_mode(self):
        return self.mode_group.checkedId()

    def apply_scale_style(self):
        self.setStyleSheet(make_app_style(self.current_scale))

    def change_scale(self, label):
        self.current_scale = STYLE_SCALES[label]
        self.apply_scale_style()
        self.update_board_button_sizes()

    def parse_board_size(self):
        text = self.input_box.text().strip()
        if not text:
            return 5
        n = int(text)
        if n <= 0:
            raise ValueError
        return n

    def create_grid(self):
        try:
            n = self.parse_board_size()
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的正整数网格大小。")
            return
        self.create_grid_with_state(n, [STATE_WHITE] * (n * n), clear_result=True)

    def create_grid_with_state(self, n, state, clear_result=False):
        if self.grid is not None:
            self.clear_existing_grid()

        self.n_value = n
        self.started = False
        self.grid_state = list(state)
        self.cell_buttons = []
        if clear_result:
            self.result_box.clear()

        self.grid = QGridLayout()
        self.grid.setSpacing(8)
        self.grid.setAlignment(Qt.AlignCenter)

        for i in range(n):
            for j in range(n):
                index = i * n + j
                button = QPushButton()
                button.setCursor(Qt.PointingHandCursor)
                button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                button.clicked.connect(lambda _, idx=index: self.toggle_state(idx))
                self.cell_buttons.append(button)
                self.grid.addWidget(button, i, j)

        board_widget = QWidget()
        board_widget.setLayout(self.grid)
        self.board_wrapper.addWidget(board_widget, alignment=Qt.AlignCenter)

        for index, cell_state in enumerate(self.grid_state):
            self.set_cell_state(index, cell_state)

        self.update_status_text("编辑中")
        self.board_meta.setText(f"{n} x {n}")
        self.update_board_hint()
        self.update_board_button_sizes()

    def clear_existing_grid(self):
        while self.board_wrapper.count():
            item = self.board_wrapper.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.grid = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_layout_mode()
        self.update_board_button_sizes()

    def update_layout_mode(self):
        if self.width() < 980:
            self.content_layout.setDirection(QBoxLayout.TopToBottom)
            self.control_panel.setMinimumWidth(0)
            self.control_panel.setMaximumWidth(16777215)
        else:
            self.content_layout.setDirection(QBoxLayout.LeftToRight)
            self.control_panel.setMinimumWidth(340)
            self.control_panel.setMaximumWidth(420)

    def update_board_button_sizes(self):
        if not self.cell_buttons or not self.n_value or self.grid is None:
            return

        available_width = max(260, self.board_container.width() - 40)
        available_height = max(260, self.board_container.height() - 40)
        usable_edge = min(available_width, available_height)
        spacing_total = max(0, (self.n_value - 1) * self.grid.spacing())
        base_min = int(42 * self.current_scale)
        base_max = int(88 * self.current_scale)
        cell_size = max(base_min, min(base_max, int((usable_edge - spacing_total) / max(1, self.n_value))))

        for button in self.cell_buttons:
            button.setFixedSize(cell_size, cell_size)

    def set_cell_state(self, index, state):
        self.grid_state[index] = state
        button = self.cell_buttons[index]
        button.setProperty("cellState", str(state))
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()
        button.setEnabled(state != STATE_BLOCKED or not self.started)

    def toggle_state(self, index):
        if not self.grid_state:
            return
        if self.started:
            self.apply_move(index)
        else:
            self.toggle_edit_state(index)

    def toggle_edit_state(self, index):
        current = self.grid_state[index]
        if self.current_mode() == 1:
            next_state = STATE_WHITE if current == STATE_BLACK else STATE_BLACK
        else:
            next_state = {
                STATE_WHITE: STATE_BLACK,
                STATE_BLACK: STATE_BLOCKED,
                STATE_BLOCKED: STATE_WHITE,
            }[current]
        self.set_cell_state(index, next_state)

    def apply_move(self, index):
        if self.grid_state[index] == STATE_BLOCKED:
            return

        n = self.n_value
        row, col = divmod(index, n)
        affected = set()

        for i in range(n):
            affected.add(row * n + i)
            affected.add(i * n + col)
        affected.add(index)

        for affected_index in affected:
            if self.grid_state[affected_index] == STATE_BLOCKED:
                continue
            next_state = STATE_WHITE if self.grid_state[affected_index] == STATE_BLACK else STATE_BLACK
            self.set_cell_state(affected_index, next_state)

    def handle_mode_change(self):
        if self.current_mode() == 1 and self.grid_state:
            for index, state in enumerate(self.grid_state):
                if state == STATE_BLOCKED:
                    self.set_cell_state(index, STATE_WHITE)
        self.update_board_hint()
        self.update_board_button_sizes()

    def update_board_hint(self):
        if not self.grid_state:
            self.board_hint.setText("先输入大小并生成棋盘，或者直接随机一局。")
            return

        if self.started:
            if self.current_mode() == 1:
                self.board_hint.setText("经典模式游玩中：点击会翻转同行、同列与自身。")
            else:
                self.board_hint.setText("异形模式游玩中：灰色格子不会参与点击和翻转。")
            return

        if self.current_mode() == 1:
            self.board_hint.setText("经典模式编辑中：点击格子切换黑白。")
        else:
            self.board_hint.setText("异形模式编辑中：点击格子按 白色 -> 深色 -> 灰色禁用 循环。")

    def start_game(self):
        if not self.grid_state:
            QMessageBox.warning(self, "提示", "请先生成棋盘。")
            return
        self.started = True
        for index, state in enumerate(self.grid_state):
            self.set_cell_state(index, state)
        self.update_status_text("游玩中")
        self.update_board_hint()

    def end_game(self):
        if not self.grid_state:
            return
        self.started = False
        for index, state in enumerate(self.grid_state):
            self.set_cell_state(index, state)
        self.update_status_text("编辑中")
        self.update_board_hint()

    def reset_board(self):
        if not self.grid_state:
            return
        self.started = False
        fill_state = STATE_WHITE
        self.grid_state = [fill_state] * len(self.grid_state)
        for index in range(len(self.grid_state)):
            self.set_cell_state(index, fill_state)
        self.result_box.clear()
        self.update_status_text("编辑中")
        self.update_board_hint()

    def randomize_current_mode(self):
        mode = self.current_mode()
        if mode == 1:
            self.randomize_classic_mode()
        else:
            self.randomize_special_mode()

    def randomize_classic_mode(self):
        self.classic_mode.setChecked(True)
        self.generate_random_level(1)

    def randomize_special_mode(self):
        self.special_mode.setChecked(True)
        self.generate_random_level(2)

    def generate_random_level(self, mode):
        try:
            n = self.parse_board_size()
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的正整数网格大小。")
            return

        if mode == 1:
            state = build_random_classic_level(n, self.rng)
        else:
            state = build_random_irregular_level(n, self.rng)

        self.create_grid_with_state(n, state, clear_result=True)
        self.update_status_text("已生成随机关卡")
        self.board_hint.setText("随机关卡已生成，可以直接开始游玩，或者先点击“计算解”查看答案。")

    def save_state(self):
        if not self.grid_state:
            QMessageBox.warning(self, "提示", "请先生成棋盘。")
            return

        mode = self.current_mode()
        if mode == 1:
            matrix = np.array([0 if value == STATE_BLOCKED else value for value in self.grid_state], dtype=int).reshape(-1, 1)
            matrix0 = matrix.reshape((self.n_value, self.n_value))
        else:
            matrix = np.array(self.grid_state, dtype=int).reshape(-1, 1)
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
