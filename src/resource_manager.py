import os
import sys
import enum


class RMType:
    Source = 'src'
    UI = 'ui'
    Icons = 'icons'
    Docs = 'docs'
    Root = ''


def path(relative_path, rm_type: RMType = None):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
        if rm_type is None:
            rm_type = RMType.Root
        if rm_type is not RMType.Root:
            base_path += os.path.sep + rm_type
    return os.path.join(base_path, relative_path)
