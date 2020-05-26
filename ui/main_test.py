# ------------------------------------------------------
# ---------------------- main_test.py ------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

import numpy as np
import random


class MatplotlibWidget(QMainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)
        
        loadUi("qt_designer.ui",self)
        
        self.setWindowTitle("PyQt5 & Matplotlib Example GUI")
        
        self.pushButton_generate_random_signal.clicked.connect(self.update_graph)
        
        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
    
    def update_graph(self):
        
        fs = 500
        f = random.randint(1, 100)
        ts = 1/fs
        length_of_signal = 100
        t = np.linspace(0, 1, length_of_signal)
        
        cosines_signal = np.cos(2*np.pi*f*t)
        sines_signal = np.sin(2*np.pi*f*t)
        
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(t, cosines_signal)
        self.MplWidget.canvas.axes.plot(t, sines_signal)
        self.MplWidget.canvas.axes.legend(('cosinus', 'sinus'),loc='upper right')
        self.MplWidget.canvas.axes.set_title('Cosinus - Sinus Signal')
        self.MplWidget.canvas.draw()


app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()
