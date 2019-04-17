from PyQt5 import uic
import fileinput

with open("mainwindow.py", "w") as f:
    uic.compileUi("mainwindow.ui", f)

with open("fitting_dialog.py", "w") as f:
    uic.compileUi("fitting_dialog.ui", f)

for line in fileinput.FileInput("mainwindow.py", inplace=1):
    if "from PyQt5 import QtCore, QtGui, QtWidgets" in line:
        line = line.replace(line, line + "\nfrom ui.matplotlibwidget import MatplotlibWidget\n\n")

    if "from mplwidget import MplWidget" in line:
        line = line.replace(line, "")

    if "self.MW_Intensity = MplWidget(self.tabIntensity)" in line:
        line = line.replace(line, "        self.MW_Intensity = MatplotlibWidget(self.tabIntensity)\n")

    if "self.MW_Lifetime = MplWidget(self.tabLifetime)" in line:
        line = line.replace(line, "        self.MW_Lifetime = MatplotlibWidget(self.tabLifetime)\n")

    if "self.MW_LifetimeInt = MplWidget(self.tabLifetime)" in line:
        line = line.replace(line, "        self.MW_LifetimeInt = MatplotlibWidget(self.tabLifetime)\n")

    if "self.MW_Spectra = MplWidget(self.tabSpectra)" in line:
        line = line.replace(line, "        self.MW_Spectra = MatplotlibWidget(self.tabSpectra)\n")

    print(line, end='')

for line in fileinput.FileInput("fitting_dialog.py", inplace=1):
    if "from PyQt5 import QtCore, QtGui, QtWidgets" in line:
        line = line.replace(line, line + "\nfrom ui.matplotlibwidget import MatplotlibWidget\n\n")

    if "from mplwidget import MplWidget" in line:
        line = line.replace(line, "")

    if "self.MW_fitparam = MplWidget(self.right_frame)" in line:
        line = line.replace(line, "        self.MW_fitparam = MatplotlibWidget(self.right_frame)\n")

    print(line, end='')
