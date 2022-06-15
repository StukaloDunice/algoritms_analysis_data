import pandas as pd
from PyQt5.QtCore import Qt
from graphviz import Digraph
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QVBoxLayout, QComboBox, QPushButton, QLineEdit)
from PyQt5.QtGui import QPixmap
from algoritms import DecisionTree as CART
from algoritms import DecisionTreeVTwo as C4_5
le = LabelEncoder()
sc = StandardScaler()

class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.min_sample_leaf = 31
        self.score = 0
        self.tree = None
        self.train = pd.read_csv("data/train3.csv")
        self.test = pd.read_csv("data/test3.csv")
        self.type_algorithm = ''
        self.preparingData()
        self.initUI()

    def initUI(self):

        self.lbl = QLabel()

        self.setGeometry(300,300, 1920, 1080)
        # верхняя горизонтальная часть
        self.vbox1 = QVBoxLayout()
        # нижняя горизонтальная часть
        self.vbox2 = QVBoxLayout()
        self.vbox2.setSpacing(10)
        # вертикальная часть
        self.hbox = QHBoxLayout()
        self.hbox.addLayout(self.vbox1,70)
        self.hbox.addLayout(self.vbox2,30)

        self.combo = QComboBox()
        self.combo.addItems(["20% выборки", "50% выборки",
                        "80% выборки"])
        self.combo.setMaximumHeight(20)
        self.vbox2.addWidget(self.combo)
        self.combo.activated[str].connect(self.onActivated)

        self.sampleEdit = QLineEdit()
        self.sampleEdit.textChanged[str].connect(self.getValueInput)
        self.vbox2.addWidget(self.sampleEdit)
        self.sampleEdit.move(35, 40)

        self.CART = QPushButton("Дерево решений CART")
        self.C4_5 = QPushButton("Дерево решений C4.5")
        self.CART.clicked.connect(self.buildDecTreeCART)
        self.C4_5.clicked.connect(self.buildDecTreeC4_5)
        self.vbox2.addWidget(self.C4_5)
        self.vbox2.addWidget(self.CART)

        self.setLayout(self.hbox)
        self.setWindowTitle('Предсказание страхования')
        self.show()

    def getValueInput(self,text):
        print(text)

    def preparingData(self):
        self.train = self.train.drop(['Unnamed: 0'], axis=1)
        self.test = self.test.drop(['Unnamed: 0'], axis=1)
        self.text_train = self.train.select_dtypes(include='object').columns
        self.float1_train = self.train.select_dtypes(exclude='object').columns
        self.text_test = self.test.select_dtypes(include='object').columns
        self.float1_test = self.test.select_dtypes(exclude='object').columns
        for col in self.text_train:
            self.train[col] = le.fit_transform(self.train[col])
        for col in self.text_test:
            self.test[col] = le.fit_transform(self.test[col])

    def onActivated(self, text):
        if text == "20% выборки":
            self.train = pd.read_csv("data/train3.csv")
            self.test = pd.read_csv("data/test3.csv")
        if text == "50% выборки":
            self.train = pd.read_csv("data/train2.csv")
            self.test = pd.read_csv("data/test2.csv")
        if text == "80% выборки":
            self.train = pd.read_csv("data/train1.csv")
            self.test = pd.read_csv("data/test1.csv")
        self.preparingData()

    def buildDecTreeC4_5(self):
        self.type_algorithm = 'C4.5'
        self.tree = C4_5.build_tree(self.train, self.min_sample_leaf)
        self.score = C4_5.hit_rate(self.tree, self.test)
        self.visualizeTree()

    def buildDecTreeCART(self):
        self.type_algorithm = 'CART'
        self.tree = CART.build_tree(self.train, self.min_sample_leaf)
        self.score = CART.hit_rate(self.tree, self.test)
        self.visualizeTree()

    def visualizeTree(self):
        self.dot = Digraph(format='png')
        self.dot.attr('node', shape='box')
        if self.type_algorithm == 'CART':
            for i in range(len(self.tree)):
                self.dot.node(str(i),
                     f'{self.tree[i].feature} <= {self.tree[i].split}\n gini = {self.tree[i].Gini}\n samples = {self.tree[i].data_index.size}\n value = {[self.tree[i].left, self.tree[i].right]} \n out = {self.tree[i].out}')
        if self.type_algorithm == 'C4.5':
            for i in range(len(self.tree)):
                self.dot.node(str(i),
                          f'{self.tree[i].feature} <= {self.tree[i].split}\n entropy = {self.tree[i].Entropy}\n samples = {self.tree[i].data.shape[0]}\n value = {[self.tree[i].left, self.tree[i].right]} \n out = {self.tree[i].out}')
        for i in range(len(self.tree)):
            self.left = self.tree[i].left
            self.right = self.tree[i].right
            if self.left != None and self.right != None:
                for j in range(self.left, self.right + 1):
                    self.dot.edge(str(i), str(j))
            elif self.left != None:
                self.dot.edge(str(i), str(self.left))
            elif self.right != None:
                self.dot.edge(str(i), str(self.right))
        self.dot.render(directory='doctest-output')

        self.pixmap = QPixmap("./doctest-output/Digraph.gv.png")
        self.pixmap = self.pixmap.scaled(1344, 920)
        self.lbl.setPixmap(self.pixmap)
        self.vbox1.addWidget(self.lbl, 1, Qt.AlignCenter)