from __future__ import annotations
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QDialog
from PyQt5 import uic
from typing import TYPE_CHECKING, List

import file_manager as fm

if TYPE_CHECKING:
    from main import MainWindow

selection_dialog_path = fm.path(name="selection_dialog.ui", file_type=fm.Type.UI)
UI_Range_Selection, _ = uic.loadUiType(selection_dialog_path)


class RangeSelectionDialog(QDialog, UI_Range_Selection):

    def __init__(self, main_window: MainWindow):
        QDialog.__init__(self)
        UI_Range_Selection.__init__(self)
        self.setupUi(self)
        self.parent = main_window

        reg_exp = QRegExp("\d+(\d*\,*\/*\s*\-*)*")
        reg_val = QRegExpValidator(reg_exp)
        self.edtSelection_Range.setValidator(reg_val)

    def get_selection(self, max_range: int = 100) -> List[int]:
        range_text = self.edtSelection_Range.text()
        range_text = range_text.replace(" ", "")
        range_text_split = range_text.split(",")
        range_indexes = list()
        if len(range_text):
            for range_section in range_text_split:
                if range_section.find("-") == -1:
                    range_indexes.append(int(range_section))
                else:
                    section_range_text = range_section.split("-")
                    if section_range_text[1] == "":
                        section_range_text[1] = str(max_range)
                    section_range_indexes = list(range(int(section_range_text[0]),
                                                       int(section_range_text[1]) + 1))
                    range_indexes.extend(section_range_indexes)

        return range_indexes

    def get_mode(self) -> List[bool]:
        mode_only = self.rdbOnly.isChecked()
        mode_add = self.rdbAdd.isChecked()
        mode_remove = self.rdbRemove.isChecked()
        mode_invert = self.rdbInvert.isChecked()

        return mode_only, mode_add, mode_remove, mode_invert
