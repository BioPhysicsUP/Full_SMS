import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMessageBox, QPushButton

from typing import Tuple


class TimedMessageBox(QMessageBox):
    def __init__(
        self,
        timeout=3,
        user_test=None,
        message_pretime=None,
        message_posttime=None,
        parent=None,
    ):
        super(TimedMessageBox, self).__init__(parent)
        self.default_button = None
        self.setWindowTitle("wait")
        self.time_to_wait = timeout
        self.user_text = user_test
        if message_pretime is None:
            message_pretime = "Wait (closing automatically in "
        if message_posttime is None:
            message_posttime = " seconds.)"
        self.message_pretime = message_pretime
        self.message_posttime = message_posttime
        self.setText(
            f"{self.user_text}\n\n{self.message_pretime}{timeout}{self.message_posttime}"
        )
        self.setStandardButtons(QMessageBox.NoButton)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.change_content)
        self.change_content()

    def setDefaultButton(self, button: QPushButton) -> None:
        self.default_button = button
        super().setDefaultButton(button)

    def setText(self, a0: str) -> None:
        self.user_text = a0
        self._new_set_text()

    def set_timeout_text(self, message_pretime: str, message_posttime: str):
        self.message_pretime = message_pretime
        self.message_posttime = message_posttime
        self._new_set_text()

    def _new_set_text(self):
        super().setText(
            f"{self.user_text}\n\n{self.message_pretime}{self.time_to_wait}{self.message_posttime}"
        )

    def exec(self) -> Tuple[int, bool]:
        self.timer.start()
        result = super().exec()
        if result == 1:
            return self.default_button, True
        else:
            return result, False

    def change_content(self):
        self.time_to_wait -= 1
        self._new_set_text()
        if self.time_to_wait <= 0:
            self.setResult(1)
            self.close()

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
