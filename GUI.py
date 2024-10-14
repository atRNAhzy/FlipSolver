import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit, QPushButton, QVBoxLayout, QMessageBox, \
    QCheckBox
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QThread, pyqtSignal
from buildMatrix import generate_matrix,generate_matrix_2
from solve_equation import gf2_gauss_jordan
import numpy as np


def output_coordinates(matrix, n):
    # 还原为n*n的方格
    grid = matrix.reshape((n, n))
    coordinates = []  # 用于保存坐标
    # 遍历网格
    for i in range(n):
        for j in range(n):
            if grid[i, j] == 1:  # 如果点的值为1
                coordinates.append((i + 1, j + 1))  # 保存坐标
    return coordinates
def find_coordinates(a, n):
    x = a // n + 1  # 行号
    y = a % n + 1   # 列号
    return (x, y)

class Worker(QThread):
    # 定义一个信号用于发送处理结果
    processed = pyqtSignal(np.ndarray)

    def __init__(self, matrix, n, matrix0, mode):
        super().__init__()
        self.matrix = matrix
        self.n_value = n
        self.matrix0 = matrix0
        self.mode = mode

    def run(self):
        if self.mode==1:
            A = generate_matrix(self.n_value)
            x, status = gf2_gauss_jordan(A, self.matrix)
            if status != '无解':
                print(status)
                coords = output_coordinates(x, self.n_value)
                for coord in coords:
                    print(f"坐标: {coord}")  # 输出坐标
            else:
                print(status)
        elif self.mode==2:
            A_1, matrix, n1,dict_index = generate_matrix_2(self.n_value,self.matrix0)
            x,status=gf2_gauss_jordan(A_1,matrix)
            if status != '无解':
                print(status)
                for index,i in enumerate(x):
                    if i==1:
                        print(find_coordinates(dict_index[index],self.n_value))
            else:
                print(status)


class GridApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.grid_state = []  # 用于存储网格状态
        self.started = False  # 标志是否开始
        self.n_value = None

    def initUI(self):
        self.setWindowTitle('点击网格')
        self.layout = QVBoxLayout()

        # 输入框和按钮
        self.inputBox = QLineEdit(self)
        self.inputBox.setPlaceholderText('输入网格大小 n')
        self.submitButton = QPushButton('生成网格', self)
        self.submitButton.clicked.connect(self.createGrid)

        self.startButton = QPushButton('开始', self)
        self.startButton.clicked.connect(self.startGame)

        self.endButton = QPushButton('结束', self)
        self.endButton.clicked.connect(self.endGame)

        self.saveButton = QPushButton('求解', self)
        self.saveButton.clicked.connect(self.saveState)

        # 添加模式选择的复选框
        self.mode1_checkbox = QCheckBox('经典模式', self)
        self.mode2_checkbox = QCheckBox('异形模式', self)

        # 布局
        self.layout.addWidget(self.inputBox)
        self.layout.addWidget(self.submitButton)
        self.layout.addWidget(self.startButton)
        self.layout.addWidget(self.endButton)
        self.layout.addWidget(self.saveButton)
        self.layout.addWidget(self.mode1_checkbox)
        self.layout.addWidget(self.mode2_checkbox)
        self.setLayout(self.layout)

        self.grid = None

    def createGrid(self):
        if self.grid:
            # 清空之前的网格
            for i in range(self.grid.count()):
                widget = self.grid.itemAt(i).widget()
                if widget:
                    widget.deleteLater()

        try:
            n = int(self.inputBox.text())
            self.n_value = n
            if n <= 0:
                raise ValueError

            self.grid = QGridLayout()
            self.grid.setSpacing(5)  # 设置网格间距
            self.grid_state = [0] * (n * n)  # 初始化状态为全白（0）

            for i in range(n):
                for j in range(n):
                    button = QPushButton()
                    button.setStyleSheet("background-color: white;")
                    button.setFixedSize(50, 50)  # 设置格子的大小
                    button.clicked.connect(lambda checked, b=button, idx=i * n + j: self.toggleState(b, idx))
                    self.grid.addWidget(button, i, j)

            # 添加网格到主布局
            self.layout.addLayout(self.grid)

        except ValueError:
            self.inputBox.setText('请输入有效的正整数')

    def startGame(self):
        """开始游戏时的操作"""
        self.started = True  # 设置开始标志

        # 如果模式2被选中，锁定非黑色的格子
        if self.mode2_checkbox.isChecked():
            self.lockNonBlackButtons()

    def endGame(self):
        self.started = False  # 设置结束标志

    def lockNonBlackButtons(self):
        """将所有非黑色的按钮变成灰色并锁定，且设置其状态为2"""
        n = self.n_value  # 网格大小
        for i in range(n):
            for j in range(n):
                button = self.grid.itemAt(i * n + j).widget()
                current_color = button.palette().button().color()
                # 如果当前格子的颜色不是黑色，将其锁定为灰色并设置状态值为2
                if current_color != QColor("black"):
                    button.setStyleSheet("background-color: gray;")  # 变为灰色
                    button.setEnabled(False)  # 禁用按钮，无法点击
                    self.grid_state[i * n + j] = 2  # 设置为2表示灰色锁定状态

    def toggleState(self, button, index):
        """切换格子的状态"""
        if self.started:
            # 获取网格的大小
            n = int(len(self.grid_state) ** 0.5)
            row, col = divmod(index, n)

            # 只有当按钮不是灰色时才响应点击事件
            current_color = button.palette().button().color()
            if current_color == QColor("gray"):  # 忽略灰色格子
                return

            # 切换点击点所在行的所有格子的状态
            for i in range(n):
                row_button = self.grid.itemAt(row * n + i).widget()
                row_color = row_button.palette().button().color()

                if row_color != QColor("gray"):  # 只切换非灰色的格子
                    row_new_color = QColor("black") if row_color != QColor("black") else QColor("white")
                    row_button.setStyleSheet(f"background-color: {row_new_color.name()};")
                    self.grid_state[row * n + i] = 1 if row_new_color == QColor("black") else 0

            # 切换点击点所在列的所有格子的状态
            for i in range(n):
                col_button = self.grid.itemAt(i * n + col).widget()
                col_color = col_button.palette().button().color()

                if col_color != QColor("gray"):  # 只切换非灰色的格子
                    col_new_color = QColor("black") if col_color != QColor("black") else QColor("white")
                    col_button.setStyleSheet(f"background-color: {col_new_color.name()};")
                    self.grid_state[i * n + col] = 1 if col_new_color == QColor("black") else 0

            # 最后单独切换点击的格子状态
            new_color = QColor("black") if current_color != QColor("black") else QColor("white")
            button.setStyleSheet(f"background-color: {new_color.name()};")
            self.grid_state[index] = 1 if new_color == QColor("black") else 0

        else:
            # 如果未开始，点击格子只改变该格子的状态
            current_color = button.palette().button().color()
            new_color = QColor("black") if current_color != QColor("black") else QColor("white")
            button.setStyleSheet(f"background-color: {new_color.name()};")
            self.grid_state[index] = 1 if new_color == QColor("black") else 0  # 更新状态列表

    def saveState(self):
        if self.mode2_checkbox.isChecked():
            self.lockNonBlackButtons()
        if self.grid_state:
            if self.mode2_checkbox.isChecked():
                mode = 2
            else:
                mode = 1
            matrix = np.array(self.grid_state).reshape(-1, 1)
            matrix0 = matrix.reshape((self.n_value, self.n_value))
            self.runWorker(matrix, self.n_value, matrix0,mode)  # 启动工作线程
        else:
            QMessageBox.warning(self, '警告', '请先生成网格！')

    def runWorker(self, matrix, n, matrix0, mode):

        self.worker = Worker(matrix, n, matrix0,mode)
        self.worker.processed.connect(self.handleProcessed)  # 连接信号
        self.worker.start()  # 启动线程

    def handleProcessed(self, result):
        QMessageBox.information(self, '处理结果', f'矩阵的和为: {result}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GridApp()
    window.show()
    sys.exit(app.exec_())
