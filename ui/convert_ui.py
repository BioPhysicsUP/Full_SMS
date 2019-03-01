from PyQt5 import uic

with open("ui/mainwindow.py", "w") as f:
	uic.compileUi("ui/mainwindow.ui", f)