from __future__ import annotations
from PyQt5.QtCore import QAbstractItemModel, Qt, QModelIndex
from PyQt5.QtGui import QPixmap, QIcon

# import smsh5
from my_logger import setup_logger
from typing import TYPE_CHECKING
import file_manager as fm

if TYPE_CHECKING:
    import smsh5

logger = setup_logger(__name__)


# class ParitcleIcons:
#     def __init__(self):


class DatasetTreeNode(object):
    """ Contains the files with their respective particles. Also seems to house the actual data objects. """

    def __init__(self, name, dataobj, datatype, checked=False) -> None:
        """
        TODO Docstring

        Parameters
        ----------
        name
        dataobj
        datatype
        """

        self._data = name
        if type(name) == tuple:
            self._data = list(name)
        if type(name) in (str, bytes) or not hasattr(name, '__getitem__'):
            self._data = [name]
            # self._data = [name, False]

        self._columncount = len(self._data)
        self._children = []
        self._parent = None
        self._row = 0

        self.datatype = datatype
        if datatype == 'dataset':
            pass

        elif datatype == 'particle':
            pass

        # self.icon = QIcon('c:\\google drive\\current_projects\\full_sms\\resources\\icons\\group-all.png')
        self.dataobj = dataobj
        self.setChecked(checked)

    def checked(self):
        """
        Appears to be used internally.

        Returns
        -------
        Returns check status.
        """
        return self._checked

    def setChecked(self, checked=True):
        self._checked = bool(checked)

    def data(self, in_column):
        """ TODO: Docstring """

        if in_column == 0:
            return self._data[in_column]
        # elif in_column >=1 and in_column <= len(self._data) and self.datatype == 'particle':
        #     return QIcon('c:\\google drive\\current_projects\\full_sms\\resources\\icons\\group-all.png')

    def columnCount(self):
        """ TODO: Docstring """

        return self._columncount

    def childCount(self):
        """ TODO: Docstring """

        return len(self._children)

    def child(self, in_row):
        """ TODO: Docstring """

        if in_row >= 0 and in_row < self.childCount():
            return self._children[in_row]

    def parent(self):
        """ TODO: Docstring """

        return self._parent

    def row(self):
        """ TODO: Docstring """

        return self._row

    def addChild(self, in_child):
        """
        TODO: Docstring

        Parameters
        ----------
        in_child
        """

        in_child._parent = self
        in_child._row = len(self._children)
        self._children.append(in_child)
        self._columncount = max(in_child.columnCount(), self._columncount)

        return in_child._row


class DatasetTreeModel(QAbstractItemModel):
    """ TODO: Docstring """

    def __init__(self):
        """ TODO: Docstring """

        QAbstractItemModel.__init__(self)
        self._root = DatasetTreeNode(None, None, None)
        # for node in in_nodes:
        #     self._root.addChild(node)
        self.none = QPixmap(fm.path('particle-none.png', fm.Type.Icons)).scaledToHeight(12)
        self.e = QPixmap(fm.path('particle-e.png', fm.Type.Icons)).scaledToHeight(12)
        self.r = QPixmap(fm.path('particle-r.png', fm.Type.Icons)).scaledToHeight(12)
        self.re = QPixmap(fm.path('particle-re.png', fm.Type.Icons)).scaledToHeight(12)
        self.rg = QPixmap(fm.path('particle-rg.png', fm.Type.Icons)).scaledToHeight(12)
        self.rge = QPixmap(fm.path('particle-rge.png', fm.Type.Icons)).scaledToHeight(12)
        self.l = QPixmap(fm.path('particle-l.png', fm.Type.Icons)).scaledToHeight(12)
        self.le = QPixmap(fm.path('particle-le.png', fm.Type.Icons)).scaledToHeight(12)
        self.rl = QPixmap(fm.path('particle-rl.png', fm.Type.Icons)).scaledToHeight(12)
        self.rle = QPixmap(fm.path('particle-rle.png', fm.Type.Icons)).scaledToHeight(12)
        self.rgl = QPixmap(fm.path('particle-rgl.png', fm.Type.Icons)).scaledToHeight(12)
        self.rgle = QPixmap(fm.path('particle-rgle.png', fm.Type.Icons)).scaledToHeight(12)

    def flags(self, index):
        # return self.flags(index) | Qt.ItemIsUserCheckable
        flags = Qt.ItemIsEnabled | Qt.ItemIsTristate | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        return flags

    def rowCount(self, in_index):
        """
        TODO: Docstring

        Parameters
        ----------
        in_index
        """

        if in_index.isValid():
            return in_index.internalPointer().childCount()
        return self._root.childCount()

    def addChild(self, in_node, in_parent=None):  #, progress_sig=None):
        """
        TODO: Docstring

        Parameters
        ----------
        in_node
        in_parent
        progress_sig
        """

        self.layoutAboutToBeChanged.emit()
        if not in_parent or not in_parent.isValid():
            parent = self._root
        else:
            parent = in_parent.internalPointer()
        row = parent.addChild(in_node)
        self.layoutChanged.emit()
        self.modelReset.emit()
        # if progress_sig is not None:
        #     progress_sig.emit()  # Increment progress bar on MainWindow GUI
        return self.index(row, 0)

    def index(self, in_row, in_column, in_parent=None):
        """
        TODO: Docstring

        Parameters
        ----------
        in_row
        in_column
        in_parent
        """

        if not in_parent or not in_parent.isValid():
            parent = self._root
        else:
            parent = in_parent.internalPointer()

        # if not QAbstractItemModel.hasIndex(self, in_row, in_column, in_parent):
        #     return QModelIndex()

        child = parent.child(in_row)
        if child:
            return QAbstractItemModel.createIndex(self, in_row, in_column, child)
        else:
            return QModelIndex()

    def parent(self, in_index):
        """
        TODO: Docstring

        Parameters
        ----------
        in_index
        """

        if in_index.isValid():
            p = in_index.internalPointer().parent()
            if p:
                return QAbstractItemModel.createIndex(self, p.row(), 0, p)
        return QModelIndex()

    def columnCount(self, in_index):
        """
        TODO: Docstring

        Parameters
        ----------
        in_index
        """

        if in_index.isValid():
            return in_index.internalPointer().columnCount()
        return self._root.columnCount()

    def get_particle(self, ind) -> smsh5.Particle:
        """
        Returns the smsh5.Particle object of the ind'th tree particle.

        Parameters
        ----------
        ind: int
            The index of the particle.

        Returns
        -------
        smsh5.Particle
        """
        return self.data(ind, Qt.UserRole)

    def data(self, in_index, role):
        """
        TODO: Docstring

        Parameters
        ----------
        in_index
        role
        """

        if not in_index.isValid():
            return None
        node = in_index.internalPointer()
        if role == Qt.DisplayRole:
            return node.dataset(in_index.column())
        if role == Qt.DecorationRole:
            if hasattr(node, 'datatype') and node.datatype == 'particle':
                p = node.dataobj
                r = p.has_levels
                g = p.has_groups
                l = p.has_fit_a_lifetime
                e = p.has_exported

                icon = None
                if not any([r, g, l, e]):
                    icon = self.none
                elif e and not any([r, g, l]):
                    icon = self.e
                elif r and not any([g, l, e]):
                    icon = self.r
                elif r and e and not any([g, l]):
                    icon = self.re
                elif r and g and not any([l, e]):
                    icon = self.rg
                elif r and g and e and not l:
                    icon = self.rge
                elif l and not any([r, g, e]):
                    icon = self.l
                elif l and e and not any([r, g]):
                    icon = self.le
                elif r and l and not any([g, e]):
                    icon = self.rl
                elif r and l and e and not g:
                    icon = self.rle
                elif r and g and l and not e:
                    icon = self.rgl
                elif all([r, g, l, e]):
                    icon = self.rgle
                return icon
        if role == Qt.UserRole:
            return node.dataobj
        if role == Qt.CheckStateRole:
            if node.checked():
                return Qt.Checked
            return Qt.Unchecked
        return None

    def setData(self, index, value, role=Qt.EditRole):

        if index.isValid():
            if role == Qt.CheckStateRole:
                node = index.internalPointer()
                node.setChecked(not node.checked())
                return True
        return False
