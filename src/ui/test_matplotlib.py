import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QPushButton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import random


class App(QMainWindow):
	
	def __init__(self):
		super().__init__()
		self.left = 10
		self.top = 10
		self.title = 'PyQt5 matplotlib example - pythonspot.com'
		self.width = 640
		self.height = 400
		self.initUI()
	
	def initUI(self):
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)
		
		m = PlotCanvas(self, width=5, height=4)
		# print(m)
		m.move(0, 0)
		
		self.show()


class PlotCanvas(FigureCanvas):
	
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi)
		# print(fig)
		self.axes = fig.add_subplot(111)
		print(self.axes)
		
		FigureCanvas.__init__(self, fig)
		# print(FigureCanvas)
		self.setParent(parent)
		
		FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.plot()
	
	def plot(self):
		data = [random.random() for i in range(25)]
		ax = self.figure.add_subplot(111)
		# print(ax)
		ax.plot(data, 'r-')
		ax.set_title('PyQt Matplotlib Example')
		self.draw()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = App()
	app.exec_()