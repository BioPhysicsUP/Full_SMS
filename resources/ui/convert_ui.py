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
        ui_path = os.getcwd() +os.sep  # +"ui"
        print(os.sep)
    if ui_py_name is None:
        ui_py_name = "mainwindow.py"
    
    with open(ui_path+os.sep+ui_py_name, "w") as f:
        uic.compileUi(ui_path+os.sep+ui_name, f)


if __name__ == '__main__':
    convert_ui('mainwindow.ui', os.getcwd(), 'mainwindow.py')
    convert_ui('fitting_dialog.ui', os.getcwd(), 'fitting_dialog.py')
