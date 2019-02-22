# ------------------------------------------------------
# -------------------- mplwidget.py --------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure


class MplWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        
        self.canvas = FigureCanvas(Figure())
        # self.canvas = Figure()

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)

        window_color = self.palette().color(QPalette.Window)
        rgbacolor = (window_color.red()/255, window_color.green()/255, window_color.blue()/255, 1)
        self.canvas.figure.patch.set_facecolor(rgbacolor)
        # self.canvas.figure.patch.set_facecolor()

        # QColorDialog.palette().Background.__getattribute__('Color')

        # QApplication.palette().__getattribute__('Background')
        # print(self.palette().__getattribute__('Background'))