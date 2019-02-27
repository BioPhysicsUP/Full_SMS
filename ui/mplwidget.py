# ------------------------------------------------------
# -------------------- mplwidget.py --------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import*

from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['axes.linewidth'] = 0.01  # set  the value globally
mpl.rcParams['savefig.dpi'] = 400
mpl.rcParams['font.size'] = 10
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['errorbar.capsize'] = 3
# mpl.rcParams['image.interpolation'] = 'bilinear'
    
class MplWidget(QWidget):
    
    def __init__(self, parent = None):

        QWidget.__init__(self, parent)
        
        self.canvas = FigureCanvas(Figure())
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)