# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------

from PyQt5.QtWidgets import*
from PyQt5.QtCore import QObject, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.axes._subplots import Axes

import numpy as np
import random

import matplotlib as mpl

from mainwidow import Ui_MainWindow


# Default settings for matplotlib plots
# *************************************
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
mpl.rcParams['savefig.dpi'] = 400
mpl.rcParams['font.size'] = 10
# mpl.rcParams['legend.fontsize'] = 'small'
# mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['lines.linewidth'] = 1.0
# mpl.rcParams['errorbar.capsize'] = 3


class MainWindow(QMainWindow, Ui_MainWindow):
	
	def __init__(self):
		
		QMainWindow.__init__(self)
		self.setupUi(self)
		
		self.setWindowTitle("Full SMS")
		
		fig_pos = [0.09, 0.12, 0.89, 0.85]  # [left, bottom, right, top]

		fig_intensity = FigureCanvas(self.MW_Intensity.canvas.figure)
		fig_intensity.axes = self.MW_Intensity.canvas.axes
		fig_intensity.axes.set_xlabel('Time (s)')
		fig_intensity.axes.set_ylabel('Bin Intensity (counts/bin)')
		fig_intensity.axes.patch.set_linewidth(0.1)
		fig_intensity.figure.tight_layout()
		fig_intensity.axes.set_position(fig_pos)
		
		fig_life_int = FigureCanvas(self.MW_LifetimeInt.canvas.figure)
		fig_life_int.figure.set_dpi(10)
		fig_life_int.axes = self.MW_LifetimeInt.canvas.axes
		fig_life_int.axes.set_xlabel('Time (s)')
		fig_life_int.axes.set_ylabel('Bin Intensity\n(counts/bin)')
		fig_life_int.figure.tight_layout()
		fig_life_int.axes.set_position([0.12, 0.2, 0.85, 0.75])

		fig_lifetime = FigureCanvas(self.MW_Lifetime.canvas.figure)
		fig_lifetime.axes = self.MW_Lifetime.canvas.axes
		fig_lifetime.axes.set_xlabel('Time (ns)')
		fig_lifetime.axes.set_ylabel('Bin frequency\n(counts/bin)')
		fig_lifetime.figure.tight_layout()
		fig_lifetime.axes.set_position([0.12, 0.22, 0.85, 0.75])
		
		fig_spectra = FigureCanvas(self.MW_Spectra.canvas.figure)
		fig_spectra.axes = self.MW_Spectra.canvas.axes
		fig_spectra.axes.set_xlabel('Time (s)')
		fig_spectra.axes.set_ylabel('Wavelength (nm)')
		fig_spectra.figure.tight_layout()
		fig_spectra.axes.set_position(fig_pos)
		
		# Connect all GUI buttons with outside class functions
		self.btnApplyBin.clicked.connect(gui_apply_bin)
		self.btnApplyBinAll.clicked.connect(gui_apply_bin_all)
		self.btnResolve.clicked.connect(gui_resolve)
		self.btnResolveAll.clicked.connect(gui_resolve_all)
		self.btnPrevLevel.clicked.connect(gui_prev_lev)
		self.btnNextLevel.clicked.connect(gui_next_lev)
		self.btnLoadIRF.clicked.connect(gui_load_irf)
		self.btnFitParameters.clicked.connect(gui_fit_param)
		self.btnFit.clicked.connect(gui_fit_current)
		self.btnFitSelected.clicked.connect(gui_fit_selected)
		self.btnFitAll.clicked.connect(gui_fit_all)
		self.btnSubBackground.clicked.connect(gui_sub_bkg)
		self.actionOpen_h5.triggered.connect(act_open_h5)
		self.actionOpen_pt3.triggered.connect(act_open_pt3)
		self.actionTrim_Dead_Traces.triggered.connect(act_trim)
	
	def get_bin(self):
		test = QSpinBox()
		test.__init__(self.spbBinSize)
		print(test.value())
		print(self.spbBinSize.value())
		
		# return self.


def gui_apply_bin():
	print("gui_apply_bin")
	mainwindow.get_bin()
	pass


def gui_apply_bin_all():
	print("gui_apply_bin_all")
	pass


def gui_resolve():
	print("gui_resolve")
	pass


def gui_resolve_all():
	print("gui_resolve_all")
	pass


def gui_prev_lev():
	print("gui_prev_lev")
	pass


def gui_next_lev():
	print("gui_next_lev")
	pass


def gui_load_irf():
	print("gui_load_irf")
	pass


def gui_fit_param():
	print("gui_fit_param")
	pass


def gui_fit_current():
	print("gui_fit_current")
	pass


def gui_fit_selected():
	print("gui_fit_selected")
	pass


def gui_fit_all():
	print("gui_fit_all")
	pass


def gui_sub_bkg():
	print("gui_sub_bkg")
	pass


def act_open_h5():
	print("act_open_h5")
	
	pass


def act_open_pt3():
	print("act_open_pt3")
	pass


def act_trim():
	print("act_trim")
	pass

app = QApplication([])
mainwindow = MainWindow()
mainwindow.show()
app.exec_()