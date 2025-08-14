import sys
from PySide6 import QtWidgets
from gui import ChatApp  # changed from .gui to gui

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ChatApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
