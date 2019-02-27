# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------

from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.axes._subplots import Axes

import numpy as np
import random

import matplotlib as mpl


# Default settings for matplotlib plots
# *************************************
mpl.rcParams['figure.dpi'] = 50
mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
mpl.rcParams['savefig.dpi'] = 400
mpl.rcParams['font.size'] = 10
# mpl.rcParams['legend.fontsize'] = 'small'
# mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['lines.linewidth'] = 1.0
# mpl.rcParams['errorbar.capsize'] = 3


class MatplotlibWidget(QMainWindow):
	
	def __init__(self):
		QMainWindow.__init__(self)
		
		loadUi("ui/mainwindow.ui", self)
		
		self.setWindowTitle("Full SMS")

		fig_pos = [0.1, 0.15, 0.85, 0.8] # [left, bottom, right, top]

		fig_intensity = FigureCanvas(self.MW_Intensity.canvas.figure)
		fig_intensity.axes = self.MW_Intensity.canvas.axes
		fig_intensity.axes.set_xlabel('Time (s)')
		fig_intensity.axes.set_ylabel('Bin Intensity (counts/bin)')
		fig_intensity.axes.patch.set_linewidth(0.1)
		fig_intensity.figure.tight_layout()
		fig_intensity.axes.set_position(fig_pos)
		# fig_intensity.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
		
		fig_life_int = FigureCanvas(self.MW_LifetimeInt.canvas.figure)
		fig_life_int.figure.set_dpi(10)
		fig_life_int.axes = self.MW_LifetimeInt.canvas.axes
		fig_life_int.axes.set_xlabel('Time (s)')
		fig_life_int.axes.set_ylabel('Bin Intensity (counts/bin)')
		fig_life_int.figure.tight_layout()
		fig_life_int.axes.set_position(fig_pos)


		fig_lifetime = FigureCanvas(self.MW_Lifetime.canvas.figure)
		fig_lifetime.axes = self.MW_Lifetime.canvas.axes
		fig_lifetime.axes.set_xlabel('Time (ns)')
		fig_lifetime.axes.set_ylabel('Bin frequency (counts/bin)')
		fig_lifetime.figure.tight_layout()
		fig_lifetime.axes.set_position(fig_pos)
		
		fig_spectra = FigureCanvas(self.MW_Spectra.canvas.figure)
		fig_spectra.axes = self.MW_Spectra.canvas.axes
		fig_spectra.axes.set_xlabel('Time (s)')
		fig_spectra.axes.set_ylabel('Wavelength (nm)')
		fig_spectra.figure.tight_layout()
		fig_spectra.axes.set_position(fig_pos)
		
		# fig_intensity.axes.set_title('Intensity')
		# data1 = [random.random() for i in range(25)]
		# data2 = [random.random() for i in range(25)]
		# fig_intensity.axes.plot(data1)
		# fig_intensity.axes.plot(data2)

		self.pushButton_generate_random_signal.clicked.connect(self.update_graph)

	# def update_graph(self):
	#     fs = 500
	#     f = random.randint(1, 100)
	#     ts = 1 / fs
	#     length_of_signal = 100
	#     t = np.linspace(0, 1, length_of_signal)
	#
	#     cosinus_signal = np.cos(2 * np.pi * f * t)
	#     sinus_signal = np.sin(2 * np.pi * f * t)
	#
	#     self.MplWidgetIntensity.canvas.axes.clear()
	#     self.MplWidgetIntensity.canvas.axes.plot(t, cosinus_signal)
	#     self.MplWidgetIntensity.canvas.axes.plot(t, sinus_signal)
	#     self.MplWidgetIntensity.canvas.axes.legend(('cosinus', 'sinus'), loc='upper right')
	#     self.MplWidgetIntensity.canvas.draw()


class PlotCanvas(FigureCanvas):
	
	def __init__(self):
		fig = Figure()
		self.axes = fig.add_subplot(111)
		
		FigureCanvas.__init__(self, fig)
		self.setParent()
		
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

app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()