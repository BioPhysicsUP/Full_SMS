import sys
from PyQt5.Qt import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
# from copy import deepcopy
# from cPickle import dumps, load, loads
# from cStringIO import StringIO

class myNode(object):
    def __init__(self, name, state, description,otro, parent=None, checked=False):

        self.name = name
        self.state = state
        self.description = description
        self.otro = otro

        self.parent = parent
        self.children = []

        self.setParent(parent)
        self.setChecked(checked)

    def checked(self):
        return self._checked

    def setChecked(self, checked=True):
        self._checked = bool(checked)

    def setParent(self, parent):
        if parent != None:
            self.parent = parent
            self.parent.appendChild(self)
        else:
            self.parent = None

    def appendChild(self, child):
        self.children.append(child)

    def childAtRow(self, row):
        return self.children[row]

    def rowOfChild(self, child):
        for i, item in enumerate(self.children):
            if item == child:
                return i
        return -1

    def removeChild(self, row):
        value = self.children[row]
        self.children.remove(value)

        return True

    def __len__(self):
        return len(self.children)


class myModel(QAbstractItemModel):

    def __init__(self, parent=None):
        super(myModel, self).__init__(parent)
        self.treeView = parent

        self.columns = 4
        self.headers = ['Directorio','Peso','Tipo','Modificado']

        # Create items
        self.root = myNode('root', 'on', 'this is root','asd', None)
        itemA = myNode('itemA', 'on', 'this is item A','dfg', self.root)
        itemB = myNode('itemB', 'on', 'this is item B','fgh', self.root)
        itemC = myNode('itemC', 'on', 'this is item C','cvb', self.root)

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.headers[section])
        return QVariant()

    def supportedDropActions(self):
        return Qt.CopyAction | Qt.MoveAction

    def flags(self, index):
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 0:
            return flags | Qt.ItemIsUserCheckable
        return flags

    def insertRow(self, row, parent):
        return self.insertRows(row, 1, parent)

    def insertRows(self, row, count, parent):
        self.beginInsertRows(parent, row, (row + (count - 1)))
        self.endInsertRows()
        return True

    def removeRow(self, row, parentIndex):
        return self.removeRows(row, 1, parentIndex)

    def removeRows(self, row, count, parentIndex):
        self.beginRemoveRows(parentIndex, row, row)
        node = self.nodeFromIndex(parentIndex)
        node.removeChild(row)
        self.endRemoveRows()

        return True

    def index(self, row, column, parent):
        node = self.nodeFromIndex(parent)
        return self.createIndex(row, column, node.childAtRow(row))

    def data(self, index, role):

        if not index.isValid():
            return None

        node = self.nodeFromIndex(index)

        if role == Qt.DisplayRole:
            if index.column() == 0:
                return QVariant(node.name)
            if index.column() == 1:
                return QVariant(node.state)
            if index.column() == 2:
                return QVariant(node.description)
            if index.column() == 3:
                return QVariant(node.otro)
        elif role == Qt.CheckStateRole:
            if index.column() == 0:
                if node.checked():
                    return Qt.Checked
                return Qt.Unchecked

        return QVariant()

    def setData(self, index, value, role=Qt.EditRole):

        if index.isValid():
            if role == Qt.CheckStateRole:
                node = index.internalPointer()
                node.setChecked(not node.checked())
                return True
        return False

    def columnCount(self, parent):
        return self.columns

    def rowCount(self, parent):
        node = self.nodeFromIndex(parent)
        if node is None:
            return 0
        return len(node)

    def parent(self, child):
        if not child.isValid():
            return QModelIndex()

        node = self.nodeFromIndex(child)

        if node is None:
            return QModelIndex()

        parent = node.parent

        if parent is None:
            return QModelIndex()

        grandparent = parent.parent
        if grandparent is None:
            return QModelIndex()
        row = grandparent.rowOfChild(parent)

        assert row != - 1
        return self.createIndex(row, 0, parent)

    def nodeFromIndex(self, index):
        return index.internalPointer() if index.isValid() else self.root

class myTreeView(QTreeView):

    def __init__(self, parent=None):
        super(myTreeView, self).__init__(parent)

        self.myModel = myModel()
        self.setModel(self.myModel)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 400)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.treeView = myTreeView(self.centralwidget)
        self.treeView.setObjectName("treeView")
        self.horizontalLayout.addWidget(self.treeView)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 600, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QApplication.translate("MainWindow",
           "MainWindow", None, QApplication.UnicodeUTF8))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())