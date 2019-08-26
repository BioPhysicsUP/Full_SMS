from PyQt5 import uic
import fileinput
import os


def convert_ui(ui_name: str = None, ui_path: str = None, ui_py_name: str = None):
    """

    Parameters
    ----------
    ui_name : str, Optional
    ui_path : str, Optional
    ui_py_name : str, Optional
    """
    if ui_name is None:
        ui_name = "mainwindow.ui"
    if ui_path is None:
        ui_path = os.getcwd()+os.sep+"ui"
    if ui_py_name is None:
        ui_py_name = "mainwindow.py"
    
    with open(ui_path+os.sep+ui_py_name, "w") as f:
        uic.compileUi(ui_path+os.sep+ui_name, f)
    
    # for line in fileinput.FileInput(ui_path + os.sep + ui_py_name, inplace=1):
    #     if "from PyQt5 import QtCore, QtGui, QtWidgets" in line:
    #         line = line.replace(line, line + "\nfrom ui.matplotlibwidget import MatplotlibWidget\n\n")
    #
    #     if "from MatplotlibWidget import MatplotlibWidget" in line:
    #         line = line.replace(line, "")
    
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
    
    # print(line, end='')


def main():
    """
    Runs convert_ui, without input parameters.
    """
    convert_ui()


if __name__ == '__main__':
    main()
