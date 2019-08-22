
from PyQt5 import uic
import fileinput
import os

with open(os.getcwd() + os.sep + "ui" + os.sep + "mainwindow.py", "w") as f:
    uic.compileUi(os.getcwd() + os.sep + "ui" + os.sep + "mainwindow.ui", f)

for line in fileinput.FileInput(os.getcwd() + os.sep + "ui" + os.sep + "mainwindow.py", inplace=1):
    if "from PyQt5 import QtCore, QtGui, QtWidgets" in line:
        line = line.replace(line, line + "\nfrom ui.matplotlibwidget import MatplotlibWidget\n\n")

    if "from MatplotlibWidget import MatplotlibWidget" in line:
        line = line.replace(line, "")

    # if "self.MW_Intensity = MplWidget(self.tabIntensity)" in line:
    #     line = line.replace(line, "        self.MW_Intensity = MatplotlibWidget(self.tabIntensity)\n")
    #
    # if "self.MW_Lifetime = MplWidget(self.tabLifetime)" in line:
    #     line = line.replace(line, "        self.MW_Lifetime = MatplotlibWidget(self.tabLifetime)\n")
    #
    # if "self.MW_LifetimeInt = MplWidget(self.tabLifetime)" in line:
    #     line = line.replace(line, "        self.MW_LifetimeInt = MatplotlibWidget(self.tabLifetime)\n")
    #
    # if "self.MW_Spectra = MplWidget(self.tabSpectra)" in line:
    #     line = line.replace(line, "        self.MW_Spectra = MatplotlibWidget(self.tabSpectra)\n")

    print(line, end='')
