# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------

from PyQt5.QtWidgets import*
from PyQt5.QtCore import QObject, pyqtSignal, QAbstractItemModel, QModelIndex, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from platform import system

from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.axes._subplots import Axes

import numpy as np
import random

import matplotlib as mpl

import dbg

import smsh5

from ui.mainwindow import Ui_MainWindow

# mpl.use("Qt5Agg")


# Default settings for matplotlib plots
# *************************************
# mpl.rcParams['figure.dpi'] = 120
# mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
# mpl.rcParams['savefig.dpi'] = 400
# mpl.rcParams['font.size'] = 10
# # mpl.rcParams['legend.fontsize'] = 'small'
# # mpl.rcParams['legend.fontsize'] = 'small'
# mpl.rcParams['lines.linewidth'] = 1.0
# # mpl.rcParams['errorbar.capsize'] = 3


class MainWindow(QMainWindow, Ui_MainWindow):
	"""
	Class for Full SMS application that returns QMainWindow object.

	This class uses a *.ui converted to a *.py script to generate gui. Be
	sure to run convert_ui.py after having made changes to mainwindow.ui.
	"""

	def __init__(self):
		"""Initialise MainWindow object.

		Creates and populates QMainWindow object as described by mainwindow.py
		as well as creates MplWidget

		"""

		self.confidence_index = {
			0: 99,
			1: 95,
			2: 90,
			3: 69
		}

		# Set defaults for figures depending on system
		if system() == "win32" or system() == "win64":
			dbg.p(debug_print="System: Windows", debug_from="Main")
			dpi = 120
			mpl.rcParams['figure.dpi'] = dpi
			mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
			mpl.rcParams['savefig.dpi'] = 400
			mpl.rcParams['font.size'] = 10
			mpl.rcParams['lines.linewidth'] = 1.0
			fig_pos = [0.09, 0.12, 0.89, 0.85]  # [left, bottom, right, top]
			fig_life_int_pos = [0.12, 0.2, 0.85, 0.75]
			fig_lifetime_pos = [0.12, 0.22, 0.85, 0.75]
		elif system() == "Darwin":
			dbg.p("System: Unix/Linus", "Main")
			dpi = 100
			mpl.rcParams['figure.dpi'] = dpi
			mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
			mpl.rcParams['savefig.dpi'] = 400
			mpl.rcParams['font.size'] = 10
			mpl.rcParams['lines.linewidth'] = 1.0
			fig_pos = [0.1, 0.12, 0.89, 0.85]  # [left, bottom, right, top]
			fig_life_int_pos = [0.17, 0.2, 0.8, 0.75]
			fig_lifetime_pos = [0.12, 0.22, 0.85, 0.75]
		else:
			dbg.p("System: Other", "Main")
			dpi = 120
			mpl.rcParams['figure.dpi'] = dpi
			mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
			mpl.rcParams['savefig.dpi'] = 400
			mpl.rcParams['font.size'] = 10
			mpl.rcParams['lines.linewidth'] = 1.0
			fig_pos = [0.09, 0.12, 0.89, 0.85]  # [left, bottom, right, top]
			fig_life_int_pos = [0.12, 0.2, 0.85, 0.75]
			fig_lifetime_pos = [0.12, 0.22, 0.85, 0.75]

		QMainWindow.__init__(self)
		self.setupUi(self)
		# print(self.MW_Intensity.figure.get_dpi())
		
		self.setWindowTitle("Full SMS")

		self.MW_Intensity.axes.set_xlabel('Time (s)')
		self.MW_Intensity.axes.set_ylabel('Bin Intensity (counts/bin)')
		self.MW_Intensity.axes.patch.set_linewidth(0.1)
		self.MW_Intensity.figure.tight_layout()
		self.MW_Intensity.axes.set_position(fig_pos)
		# print(self.MW_Intensity.figure.get_dpi())

		self.MW_LifetimeInt.axes.set_xlabel('Time (s)')
		self.MW_LifetimeInt.axes.set_ylabel('Bin Intensity\n(counts/bin)')
		self.MW_LifetimeInt.figure.tight_layout()
		self.MW_LifetimeInt.axes.set_position(fig_life_int_pos)

		self.MW_Lifetime.axes.set_xlabel('Time (ns)')
		self.MW_Lifetime.axes.set_ylabel('Bin frequency\n(counts/bin)')
		self.MW_Lifetime.figure.tight_layout()
		self.MW_Lifetime.axes.set_position(fig_lifetime_pos)

		self.MW_Spectra.axes.set_xlabel('Time (s)')
		self.MW_Spectra.axes.set_ylabel('Wavelength (nm)')
		self.MW_Spectra.figure.tight_layout()
		self.MW_Spectra.axes.set_position(fig_pos)
		
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

		# Connect the tree selecttion to data display
		self.treeViewParticles.currentChanged.connect(display_data)

	def get_bin(self):
		"""Returns current GUI value for bin size in ms."""
		return self.spbBinSize.value()

	def get_gui_confidence(self):
		"""Return current GUI value for confidence percentage."""
		return [self.cmbConfIndex.currentIndex(), self.confidence_index[self.cmbConfIndex.currentIndex()]]


class CustomNode(object):
	def __init__(self, in_data):
		self._data = in_data
		if type(in_data) == tuple:
			self._data = list(in_data)
		if type(in_data) in (str, bytes) or not hasattr(in_data, '__getitem__'):
			self._data = [in_data]

		self._columncount = len(self._data)
		self._children = []
		self._parent = None
		self._row = 0

	def data(self, in_column):
		if in_column >= 0 and in_column < len(self._data):
			return self._data[in_column]

	def columnCount(self):
		return self._columncount

	def childCount(self):
		return len(self._children)

	def child(self, in_row):
		if in_row >= 0 and in_row < self.childCount():
			return self._children[in_row]

	def parent(self):
		return self._parent

	def row(self):
		return self._row

	def addChild(self, in_child):
		in_child._parent = self
		in_child._row = len(self._children)
		self._children.append(in_child)
		self._columncount = max(in_child.columnCount(), self._columncount)


class CustomModel(QAbstractItemModel):
	def __init__(self, in_nodes):
		QAbstractItemModel.__init__(self)
		self._root = CustomNode(None)
		for node in in_nodes:
			self._root.addChild(node)

	def rowCount(self, in_index):
		if in_index.isValid():
			return in_index.internalPointer().childCount()
		return self._root.childCount()

	def addChild(self, in_node, in_parent):
		if not in_parent or not in_parent.isValid():
			parent = self._root
		else:
			parent = in_parent.internalPointer()
		parent.addChild(in_node)

	def index(self, in_row, in_column, in_parent=None):
		if not in_parent or not in_parent.isValid():
			parent = self._root
		else:
			parent = in_parent.internalPointer()

		if not QAbstractItemModel.hasIndex(self, in_row, in_column, in_parent):
			return QModelIndex()

		child = parent.child(in_row)
		if child:
			return QAbstractItemModel.createIndex(self, in_row, in_column, child)
		else:
			return QModelIndex()

	def parent(self, in_index):
		if in_index.isValid():
			p = in_index.internalPointer().parent()
			if p:
				return QAbstractItemModel.createIndex(self, p.row(), 0, p)
		return QModelIndex()

	def columnCount(self, in_index):
		if in_index.isValid():
			return in_index.internalPointer().columnCount()
		return self._root.columnCount()

	def data(self, in_index, role):
		if not in_index.isValid():
			return None
		node = in_index.internalPointer()
		if role == Qt.DisplayRole:
			return node.data(in_index.column())
		return None

def gui_apply_bin():
	print("gui_apply_bin")
	print(main_window.get_bin())
	pass


def gui_apply_bin_all():
	print("gui_apply_bin_all")
	pass


def gui_resolve():
	print("gui_resolve")
	print(main_window.get_gui_confidence())
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
	fname = QFileDialog.getOpenFileName(main_window, 'Open HDF5 file', '', "HDF5 files (*.h5)")
	print(fname)
	dataset = smsh5.H5dataset(fname[0])
	items = []
	for particle in dataset.particles:
		items.append(CustomNode(particle.name))
	main_window.treeViewParticles.setModel(CustomModel(items))


def act_open_pt3():
	print("act_open_pt3")
	pass


def act_trim():
	print("act_trim")
	pass

def display_data():
    pass


app = QApplication([])
dbg.p(debug_print='App created', debug_from='Main')
main_window = MainWindow()
dbg.p(debug_print='Main Window created', debug_from='Main')
main_window.show()
# print(main_window.f)
dbg.p(debug_print='Main Window shown', debug_from='Main')
# main_window.MW_Intensity.figure.set_dpi(100)
# main_window.MW_Intensity.draw()
# print(main_window.MW_Intensity.figure.get_dpi())
app.exec_()
dbg.p(debug_print='App excuted', debug_from='Main')
