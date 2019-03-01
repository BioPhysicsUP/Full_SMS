from PyQt5 import uic

with open("ui/mainwidow.py", "w") as f:
	uic.compileUi("ui/mainwindow.ui", f)