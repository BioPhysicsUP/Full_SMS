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
    """Contains the files with their respective particles. Also seems to house the actual data objects."""

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
        if type(name) in (str, bytes) or not hasattr(name, "__getitem__"):
            self._data = [name]
            # self._data = [name, False]

        self._columncount = len(self._data)
        self._children = []
        self._parent = None
        self._row = 0

        self.datatype = datatype
        if datatype == "dataset":
            pass

        elif datatype == "particle":
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
        """
        Return the data from a provided index

        Parameters
        ----------
        in_column: Index of column

        Returns
        -------
        data
        """

        if in_column == 0:
            return self._data[in_column]
        # elif in_column >=1 and in_column <= len(self._data) and self.datatype == 'particle':
        #     return QIcon('c:\\google drive\\current_projects\\full_sms\\resources\\icons\\group-all.png')

    def columnCount(self):
        """
        Returns the number of columns within the node.

        Returns
        -------
        int
        """

        return self._columncount

    def childCount(self):
        """
        Returns the number of children within the node.

        Returns
        -------
        int
        """

        return len(self._children)

    def child(self, in_row):
        """
        Returns the child of the provided row index.

        Parameters
        ----------
        in_row: Row index

        Returns
        -------
        child
        """

        if in_row >= 0 and in_row < self.childCount():
            return self._children[in_row]

    def parent(self):
        """
        Returns the parent of this node.

        Returns
        -------
        parent
        """

        return self._parent

    def row(self):
        """
        Returns the row of this node.

        Returns
        -------
        row
        """

        return self._row

    def addChild(self, in_child):
        """
        Add a child at the provided index.

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
    """
    An item model for the dataset tree view.
    """

    def __init__(self):

        QAbstractItemModel.__init__(self)
        self._root = DatasetTreeNode(None, None, None)
        # for node in in_nodes:
        #     self._root.addChild(node)
        height = 16

        self.none = QPixmap(fm.path("particle-none.png", fm.Type.Icons)).scaledToHeight(height)

        self.r = QPixmap(fm.path("particle-r.png", fm.Type.Icons)).scaledToHeight(height)
        self.l = QPixmap(fm.path("particle-l.png", fm.Type.Icons)).scaledToHeight(height)
        self.a = QPixmap(fm.path("particle-a.png", fm.Type.Icons)).scaledToHeight(height)
        self.e = QPixmap(fm.path("particle-e.png", fm.Type.Icons)).scaledToHeight(height)

        self.rl = QPixmap(fm.path("particle-rl.png", fm.Type.Icons)).scaledToHeight(height)
        self.rg = QPixmap(fm.path("particle-rg.png", fm.Type.Icons)).scaledToHeight(height)
        self.ra = QPixmap(fm.path("particle-ra.png", fm.Type.Icons)).scaledToHeight(height)
        self.re = QPixmap(fm.path("particle-re.png", fm.Type.Icons)).scaledToHeight(height)

        self.la = QPixmap(fm.path("particle-la.png", fm.Type.Icons)).scaledToHeight(height)
        self.le = QPixmap(fm.path("particle-le.png", fm.Type.Icons)).scaledToHeight(height)

        self.ae = QPixmap(fm.path("particle-ae.png", fm.Type.Icons)).scaledToHeight(height)

        self.rlg = QPixmap(fm.path("particle-rlg.png", fm.Type.Icons)).scaledToHeight(height)
        self.rla = QPixmap(fm.path("particle-rla.png", fm.Type.Icons)).scaledToHeight(height)
        self.rle = QPixmap(fm.path("particle-rle.png", fm.Type.Icons)).scaledToHeight(height)

        self.rga = QPixmap(fm.path("particle-rga.png", fm.Type.Icons)).scaledToHeight(height)
        self.rge = QPixmap(fm.path("particle-rge.png", fm.Type.Icons)).scaledToHeight(height)

        self.rae = QPixmap(fm.path("particle-rae.png", fm.Type.Icons)).scaledToHeight(height)
        self.lae = QPixmap(fm.path("particle-lae.png", fm.Type.Icons)).scaledToHeight(height)

        self.rlga = QPixmap(fm.path("particle-rlga.png", fm.Type.Icons)).scaledToHeight(height)
        self.rlge = QPixmap(fm.path("particle-rlge.png", fm.Type.Icons)).scaledToHeight(height)

        self.rlae = QPixmap(fm.path("particle-rlae.png", fm.Type.Icons)).scaledToHeight(height)
        self.rgae = QPixmap(fm.path("particle-rgae.png", fm.Type.Icons)).scaledToHeight(height)

        self.rlgae = QPixmap(fm.path("particle-rlgae.png", fm.Type.Icons)).scaledToHeight(height)

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

    def addChild(self, in_node, in_parent=None):  # , progress_sig=None):
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
            return node.data(in_index.column())
        if role == Qt.DecorationRole:
            if hasattr(node, "datatype") and node.datatype == "particle":
                p = node.dataobj
                r = p.has_levels
                g = p.has_groups or p.has_global_grouping
                l = p.has_fit_a_lifetime
                a = p.has_corr
                e = p.has_exported

                icon = None
                if not any([r, g, l, a, e]):
                    icon = self.none
                elif e and not any([r, g, l, a]):
                    icon = self.e
                elif r and not any([g, l, a, e]):
                    icon = self.r
                elif all([r, e]) and not any([g, l, a]):
                    icon = self.re
                elif all([r, g]) and not any([l, a, e]):
                    icon = self.rg
                elif all([r, g, e]) and not any([l, a]):
                    icon = self.rge
                elif l and not any([r, g, a, e]):
                    icon = self.l
                elif a and not any([r, g, l, e]):
                    icon = self.a
                elif all([l, e]) and not any([r, g, a]):
                    icon = self.le
                elif all([r, l]) and not any([g, e, a]):
                    icon = self.rl
                elif all([r, l, e]) and not any([g, a]):
                    icon = self.rle
                elif all([r, g, l]) and not any([a, e]):
                    icon = self.rlg
                elif all([e, a]) and not any([r, g, l]):
                    icon = self.ae
                elif all([r, a]) and not any([g, l, e]):
                    icon = self.ra
                elif all([r, e, a]) and not any([g, l]):
                    icon = self.rae
                elif all([r, g, a]) and not any([l, e]):
                    icon = self.rga
                elif all([r, g, e]) and not any([l, a]):
                    icon = self.rge
                elif all([l, a]) and not any([r, g, a, e]):
                    icon = self.la
                elif all([l, a, e]) and not any([r, g, a]):
                    icon = self.lae
                elif all([r, l, a]) and not any([g, e, a]):
                    icon = self.rla
                elif all([r, l, a, e]) and not any([g, a]):
                    icon = self.rlae
                elif all([r, g, l, a]) and not e:
                    icon = self.rlga
                elif all([r, g, l, e]) and not a:
                    icon = self.rlge
                elif all([r, g, l, a, e]):
                    icon = self.rlgae
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
                self.dataChanged.emit(index, index)
                return True
        return False
