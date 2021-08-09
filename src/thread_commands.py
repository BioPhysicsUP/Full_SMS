from __future__ import  annotations
from enum import IntEnum, auto


class ProgressCmd(IntEnum):
    SetMax = auto()
    AddMax = auto()
    Single = auto()
    Step = auto()
    SetValue = auto()
    Complete = auto()


class StatusCmd(IntEnum):
    ShowMessage = auto()
    Reset = auto()