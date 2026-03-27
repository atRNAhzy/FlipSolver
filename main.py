import sys

from PyQt5.QtWidgets import QApplication

from flipsolver.gui import GridApp


def main():
    app = QApplication(sys.argv)
    window = GridApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
