import math
import time

import pandas as pd
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QSize, QRectF
from graphviz import Digraph
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QVBoxLayout, QComboBox, QPushButton, QLineEdit, QSizePolicy,
                             QSpacerItem, QGraphicsScene, QGraphicsView)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QBrush
from algoritms import DecisionTreeCART as CART
from algoritms import DecisionTreeVTwo as C4_5
from algoritms import RandomForestCART as RF
from algoritms import RandomForestC45 as RF4_5
le = LabelEncoder()
sc = StandardScaler()

class MainWindow(QWidget):

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

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)
        self.setGeometry(0,0, 1080, 720)
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

        self.RandForestC4_5 = QPushButton("Случайный лес C4.5")
        self.RandForestC4_5.clicked.connect(self.buildRandomForestC4_5)
        self.vbox2.addWidget(self.RandForestC4_5)

        self.vbox2.addStretch(1)
        self.algoritm = QLabel('Алгоритм: ')
        self.algoritm.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        self.vbox2.addWidget(self.algoritm)
        self.scoreLabel = QLabel('Точность классификации модели: ')
        self.scoreLabel.setFont(QtGui.QFont("Times", 11, QtGui.QFont.Bold))
        self.vbox2.addWidget(self.scoreLabel)

        self.timeCreateLabel = QLabel('Время построения модели: ')
        self.timeCreateLabel.setFont(QtGui.QFont("Times", 11, QtGui.QFont.Bold))
        self.vbox2.addWidget(self.timeCreateLabel)

        self.timeClassificationLabel = QLabel('Время классификации тестовой выборки: ')
        self.timeClassificationLabel.setFont(QtGui.QFont("Times", 11, QtGui.QFont.Bold))
        self.vbox2.addWidget(self.timeClassificationLabel)

        self.vbox2.addStretch(5)

        self.setLayout(self.hbox)
        self.setWindowTitle('Прогнозирование рисков в области страхования')
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
        self.algoritm.setText('Алгоритм: Дерево решений C4.5')
        self.scoreLabel.setText('Точность классификации модели:\n' + str(self.score))
        self.timeCreateLabel.setText('Время построения модели:\n' + str(self.t2 - self.t1) + ' секунд')
        self.timeClassificationLabel.setText('Время классификации тестовой выборки:\n' + str(self.t3 - self.t2) + ' секунд')

    def buildDecTreeCART(self):
        self.type_algorithm = 'CART'
        self.t1 = time.time()
        self.tree = CART.build_tree(self.train, self.min_sample_leaf)
        self.t2 = time.time()
        self.score = CART.hit_rate(self.tree, self.test)
        self.t3 = time.time()
        self.visualizeTree()
        self.algoritm.setText('Алгоритм: Дерево решений CART')
        self.scoreLabel.setText('Точность классификации модели:\n' + str(self.score))
        self.timeCreateLabel.setText('Время построения модели:\n' + str(self.t2 - self.t1) + ' секунд')
        self.timeClassificationLabel.setText('Время классификации тестовой выборки:\n' + str(self.t3 - self.t2) + ' секунд')

    def buildRandomForestC4_5(self):
        self.type_algorithm = 'RF'
        self.t1 = time.time()
        self.tree = RF4_5.RandomForest(self.train, self.n_trees, self.min_sample_leaf, self.ip, self.jp)
        self.t2 = time.time()
        self.score = RF4_5.hit_rate(self.tree, self.test)
        self.t3 = time.time()
        self.visualizeTree()
        self.algoritm.setText('Алгоритм: Случайный лес C4.5')
        self.scoreLabel.setText('Точность классификации модели:\n' + str(self.score))
        self.timeCreateLabel.setText('Время построения модели:\n' + str(self.t2 - self.t1) + ' секунд')
        self.timeClassificationLabel.setText(
            'Время классификации тестовой выборки:\n' + str(self.t3 - self.t2) + ' секунд')

    def buildRandomForestCART(self):
        self.type_algorithm = 'RF'
        self.t1 = time.time()
        self.tree = RF.RandomForest(self.train, self.n_trees,self.min_sample_leaf,self.ip, self.jp)
        self.t2 = time.time()
        self.score = RF.hit_rate(self.tree, self.test)
        self.t3 = time.time()
        self.visualizeTree()
        self.algoritm.setText('Алгоритм: Случайный лес CART')
        self.scoreLabel.setText('Точность классификации модели:\n' + str(self.score))
        self.timeCreateLabel.setText('Время построения модели:\n' + str(self.t2 - self.t1) + ' секунд')
        self.timeClassificationLabel.setText(
            'Время классификации тестовой выборки:\n' + str(self.t3 - self.t2) + ' секунд')

    def visualizeTree(self):
        predict = {0: 0, 1: 0}
        self.dot = Digraph(format='png')
        self.dot.attr('node', shape='box',fontsize='14')
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
                for oneNode in self.tree[i]:
                    if oneNode.out != None:
                        predict[oneNode.out] += 1
                # if i % 10 == 0:

                self.dot.node(str(i),str(max(list(predict.items()), key=lambda i: i[1])[0]))
                self.dot.edge(str(-1), str(i))
                for key in predict:
                    predict[key] = 0
        self.dot = self.dot.unflatten(stagger=15)
        self.dot.render(directory='../doctest-output')
        self.scene.clear()
        self.pixmap = QPixmap("../doctest-output/Digraph.gv.png")
        # self.pixmap = self.pixmap.scaled(1344, 920)
        self.scene.addPixmap(self.pixmap)
        self.view.setFixedSize(1344,920)
        # self.view.fitInView(QRectF(0, 0, 1344, 920), Qt.KeepAspectRatio)
        self.vbox1.addWidget(self.view, 1, Qt.AlignCenter)