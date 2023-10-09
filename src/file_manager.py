import os
import sys
from enum import Enum, auto
from my_logger import setup_logger

"""
Example Structure
*****************

Project_Root/
├── src/
│   └── main.py
|   ├── ...
|   ├── resources/
|   │   └── ui/
|   │   │   └── main_window.ui
|   │   └── icons/
|   │   │   └── app.ico
|   │   └── data/
|   │       └── file.dat
└── docs/
    └── readme.md

"""

logger = setup_logger(__name__)


class Type(Enum):
    ProjectRoot = auto()
    ResourcesRoot = auto()
    Source = auto()
    UI = auto()
    Icons = auto()
    Docs = auto()
    Data = auto()


def get_path_type_str(path_enum: Type):
    if path_enum == Type.ProjectRoot:
        return ""

    resources_folder_name = "resources"
    if path_enum == Type.ResourcesRoot:
        return resources_folder_name
    elif path_enum == Type.Source:
        return "src"
    elif path_enum == Type.Docs:
        return "docs"
    elif path_enum == Type.UI:
        return os.path.join(resources_folder_name, "ui")
    elif path_enum == Type.Icons:
        return os.path.join(resources_folder_name, "icons")
    elif path_enum == Type.Data:
        return os.path.join(resources_folder_name, "data")


def path(name: str, file_type: Type = None, custom_folder: str = None):
    """Get absolute path to resource, works for dev and for PyInstaller"""

    base_path = os.path.abspath(os.path.dirname(__file__))
    try:
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
    except:
        pass

    try:
        if file_type is None:
            file_type = Type.ProjectRoot
        else:
            assert type(file_type) is Type, "Provided file_type is not of type FileType"
        if file_type is not Type.ProjectRoot:
            base_path += os.path.sep + get_path_type_str(file_type)
        if custom_folder:
            assert (
                type(custom_folder) is str
            ), "Provided custom_folder is not of type str"
            base_path += os.path.sep + custom_folder
        return os.path.join(base_path, name)
    except Exception as err:
        logger.error(err, exc_info=True)


def folder_path(folder_name: str = None, resource_type: Type = None):
    return path(name="", file_type=resource_type, custom_folder=folder_name)
