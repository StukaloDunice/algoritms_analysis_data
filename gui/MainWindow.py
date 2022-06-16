import time

import pandas as pd
from PyQt5.QtCore import Qt
from graphviz import Digraph
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QVBoxLayout, QComboBox, QPushButton, QLineEdit, QSizePolicy,
                             QSpacerItem)
from PyQt5.QtGui import QPixmap
from algoritms import DecisionTree as CART
from algoritms import DecisionTreeVTwo as C4_5
from algoritms import RandomForest as RF
le = LabelEncoder()
sc = StandardScaler()

class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.min_sample_leaf = 31
        self.score = 0
        self.tree = None
        self.n_trees = 30
        self.ip = 0.85
        self.jp = 0.7
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
        # вертикальная часть
        self.hbox = QHBoxLayout()
        self.hbox.addLayout(self.vbox1,70)
        self.hbox.addLayout(self.vbox2,30)

        self.combo = QComboBox()
        self.combo.addItems(["20% обучающей выборки", "50% обучающей выборки", "80% обучающей выборки"])
        self.combo.activated[str].connect(self.onActivated)
        self.vbox2.addWidget(self.combo)

        self.lbl1 = QLabel('Минимальное количество объектов в узле:')
        self.vbox2.addWidget(self.lbl1)
        self.sampleEdit = QLineEdit()
        self.sampleEdit.setText(str(self.min_sample_leaf))
        self.sampleEdit.textChanged[str].connect(self.getMinSampleLeafInput)
        self.vbox2.addWidget(self.sampleEdit)

        self.CART = QPushButton("Дерево решений CART")
        self.C4_5 = QPushButton("Дерево решений C4.5")
        self.CART.clicked.connect(self.buildDecTreeCART)
        self.C4_5.clicked.connect(self.buildDecTreeC4_5)
        self.vbox2.addWidget(self.C4_5)
        self.vbox2.addWidget(self.CART)

        self.lbl2 = QLabel('Количество деревьев решений в случайном лесе:')
        self.vbox2.addWidget(self.lbl2)
        self.nTreesEdit = QLineEdit()
        self.nTreesEdit.setText(str(self.n_trees))
        self.nTreesEdit.textChanged[str].connect(self.getNTreesInput)
        self.vbox2.addWidget(self.nTreesEdit)

        self.RandForest = QPushButton("Случайный лес CART")
        self.RandForest.clicked.connect(self.buildRandomForestCART)
        self.vbox2.addWidget(self.RandForest)

        self.vbox2.addStretch(1)

        self.scoreLabel = QLabel('Точность классификации модели: ')
        self.vbox2.addWidget(self.scoreLabel)

        self.timeCreateLabel = QLabel('Время построения модели: ')
        self.vbox2.addWidget(self.timeCreateLabel)

        self.timeClassificationLabel = QLabel('Время классификации тестовой выборки: ')
        self.vbox2.addWidget(self.timeClassificationLabel)

        self.vbox2.addStretch(5)

        self.setLayout(self.hbox)
        self.setWindowTitle('Предсказание страхования')
        self.show()

    def getMinSampleLeafInput(self,text):
        try:
            self.min_sample_leaf = int(text)
        except ValueError:
            self.min_sample_leaf = 31

    def getNTreesInput(self,text):
        try:
            self.n_trees = int(text)
        except ValueError:
            self.n_trees = 30

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
        if text == "20% обучающей выборки":
            self.train = pd.read_csv("data/train3.csv")
            self.test = pd.read_csv("data/test3.csv")
        elif text == "50% обучающей выборки":
            self.train = pd.read_csv("data/train2.csv")
            self.test = pd.read_csv("data/test2.csv")
        elif text == "80% обучающей выборки":
            self.train = pd.read_csv("data/train1.csv")
            self.test = pd.read_csv("data/test1.csv")
        self.preparingData()

    def buildDecTreeC4_5(self):
        self.type_algorithm = 'C4.5'
        self.t1 = time.time()
        self.tree = C4_5.build_tree(self.train, self.min_sample_leaf)
        self.t2 = time.time()
        self.score = C4_5.hit_rate(self.tree, self.test)
        self.t3 = time.time()
        self.visualizeTree()
        self.scoreLabel.setText('Точность классификации модели: ' + str(self.score))
        self.timeCreateLabel.setText('Время построения модели: ' + str(self.t2 - self.t1) + ' секунд')
        self.timeClassificationLabel.setText('Время классификации тестовой выборки: ' + str(self.t3 - self.t2) + ' секунд')

    def buildDecTreeCART(self):
        self.type_algorithm = 'CART'
        self.t1 = time.time()
        self.tree = CART.build_tree(self.train, self.min_sample_leaf)
        self.t2 = time.time()
        self.score = CART.hit_rate(self.tree, self.test)
        self.t3 = time.time()
        self.visualizeTree()
        self.scoreLabel.setText('Точность классификации модели: ' + str(self.score))
        self.timeCreateLabel.setText('Время построения модели: ' + str(self.t2 - self.t1) + ' секунд')
        self.timeClassificationLabel.setText('Время классификации тестовой выборки: ' + str(self.t3 - self.t2) + ' секунд')

    def buildRandomForestCART(self):
        self.type_algorithm = 'RF'
        self.t1 = time.time()
        self.tree = RF.RandomForest(self.train, self.n_trees,self.min_sample_leaf,self.ip, self.jp)
        print(len(self.tree))
        self.t2 = time.time()
        self.score = RF.hit_rate(self.tree, self.test)
        self.t3 = time.time()
        self.visualizeTree()
        self.scoreLabel.setText('Точность классификации модели: ' + str(self.score))
        self.timeCreateLabel.setText('Время построения модели: ' + str(self.t2 - self.t1) + ' секунд')
        self.timeClassificationLabel.setText(
            'Время классификации тестовой выборки: ' + str(self.t3 - self.t2) + ' секунд')

    def visualizeTree(self):
        self.dot = Digraph(format='png')
        self.dot.attr('node', shape='box',fontsize='12')
        if self.type_algorithm == 'CART':
            for i in range(len(self.tree)):
                self.dot.node(str(i),
                     f'{self.tree[i].feature} <= {self.tree[i].split}\n gini = {self.tree[i].Gini}\n samples = {self.tree[i].data_index.size}\n out = {self.tree[i].out}')
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
        if self.type_algorithm == 'C4.5':
            for i in range(len(self.tree)):
                self.dot.node(str(i),
                          f'{self.tree[i].feature} <= {self.tree[i].split}\n entropy = {self.tree[i].Entropy}\n samples = {self.tree[i].data.shape[0]}\n out = {self.tree[i].out}')
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
        if self.type_algorithm == 'RF':
            self.dot.node(str(-1), "Тестовый набор данных")
            self.dot.attr('node', shape='circle')
            for i in range(len(self.tree)):
                self.dot.node(str(i),str(i))
                self.dot.edge(str(-1), str(i))
        self.dot.render(directory='doctest-output')

        self.pixmap = QPixmap("./doctest-output/Digraph.gv.png")
        self.pixmap = self.pixmap.scaled(1344, 920)
        self.lbl.setPixmap(self.pixmap)
        self.vbox1.addWidget(self.lbl, 1, Qt.AlignCenter)