import DecisionTree as CART
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from graphviz import Digraph
import sys
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QApplication)
from PyQt5.QtGui import QPixmap

le = LabelEncoder()
sc = StandardScaler()

if __name__ == "__main__":
    dot = Digraph()
    train = pd.read_csv("data/train1.csv")
    test = pd.read_csv("data/test1.csv")
    train = train.drop(['Unnamed: 0'], axis=1)
    test = test.drop(['Unnamed: 0'], axis=1)
    text_train = train.select_dtypes(include='object').columns
    float1_train = train.select_dtypes(exclude='object').columns
    text_test = test.select_dtypes(include='object').columns
    float1_test = test.select_dtypes(exclude='object').columns
    for col in text_train:
        train[col] = le.fit_transform(train[col])
    for col in text_test:
        test[col] = le.fit_transform(test[col])

    dot2 = Digraph()
    train2 = pd.read_csv("data/train2.csv")
    test2 = pd.read_csv("data/test2.csv")
    train2 = train2.drop(['Unnamed: 0'], axis=1)
    test2 = test2.drop(['Unnamed: 0'], axis=1)
    text_train2 = train2.select_dtypes(include='object').columns
    float1_train2 = train2.select_dtypes(exclude='object').columns
    text_test2 = test2.select_dtypes(include='object').columns
    float1_test2 = test2.select_dtypes(exclude='object').columns
    for col in text_train2:
        train2[col] = le.fit_transform(train2[col])
    for col in text_test2:
        test2[col] = le.fit_transform(test2[col])

    dot3 = Digraph(format='png')
    train3 = pd.read_csv("data/train3.csv")
    test3 = pd.read_csv("data/test3.csv")
    train3 = train3.drop(['Unnamed: 0'], axis=1)
    test3 = test3.drop(['Unnamed: 0'], axis=1)
    text_train3 = train3.select_dtypes(include='object').columns
    float1_train3 = train3.select_dtypes(exclude='object').columns
    text_test3 = test3.select_dtypes(include='object').columns
    float1_test3 = test3.select_dtypes(exclude='object').columns
    for col in text_train3:
        train3[col] = le.fit_transform(train3[col])
    for col in text_test3:
        test3[col] = le.fit_transform(test3[col])

    t1 = time.time()
    min_sample_leaf = 31
    tree = CART.build_tree(train ,min_sample_leaf)
    t2 = time.time()
    score = CART.hit_rate(tree, test)
    t3 = time.time()
    for i in range(len(tree)):
        dot.node(str(i),
                 f'{tree[i].feature} <= {tree[i].split}\n gini = {tree[i].Gini}\n samples = {tree[i].data_index.size}\n value = {[tree[i].left, tree[i].right]} \n out = {tree[i].out}')
    for i in range(len(tree)):
        left = tree[i].get_left()
        right = tree[i].get_right()
        if left != None and right != None:
            for j in range(left, right + 1):
                dot.edge(str(i),str(j))
        elif left != None:
            dot.edge(str(i), str(left))
        elif right != None:
            dot.edge(str(i),str(right))
    dot.render(directory='doctest-output')
    print('Время построения дерева решений равно：%f'%(t2-t1))
    print('Время классификации тестовой выборки равно：%f'%(t3-t2))
    print('Точность классификации：%f'%score)
    print('Параметр установленный на min_sample_leaf：%d'%min_sample_leaf)


    t1 = time.time()
    min_sample_leaf = 31
    tree = CART.build_tree(train2, min_sample_leaf)
    t2 = time.time()
    score = CART.hit_rate(tree, test2)
    t3 = time.time()
    for i in range(len(tree)):
        dot2.node(str(i),
                 f'{tree[i].feature} <= {tree[i].split}\n gini = {tree[i].Gini}\n samples = {tree[i].data_index.size}\n value = {[tree[i].left, tree[i].right]} \n out = {tree[i].out}')
    for i in range(len(tree)):
        left = tree[i].get_left()
        right = tree[i].get_right()
        if left != None and right != None:
            for j in range(left, right + 1):
                dot2.edge(str(i), str(j))
        elif left != None:
            dot2.edge(str(i), str(left))
        elif right != None:
            dot2.edge(str(i), str(right))
    dot2.render(directory='doctest-output')
    print('Время построения дерева решений равно：%f' % (t2 - t1))
    print('Время классификации тестовой выборки равно：%f' % (t3 - t2))
    print('Точность классификации：%f' % score)
    print('Параметр установленный на min_sample_leaf：%d' % min_sample_leaf)


    t1 = time.time()
    min_sample_leaf = 31
    tree = CART.build_tree(train3, min_sample_leaf)
    t2 = time.time()
    score = CART.hit_rate(tree, test3)
    t3 = time.time()
    for i in range(len(tree)):
        dot3.node(str(i),
                  f'{tree[i].feature} <= {tree[i].split}\n gini = {tree[i].Gini}\n samples = {tree[i].data_index.size}\n value = {[tree[i].left, tree[i].right]} \n out = {tree[i].out}')
    for i in range(len(tree)):
        left = tree[i].get_left()
        right = tree[i].get_right()
        if left != None and right != None:
            for j in range(left, right + 1):
                dot3.edge(str(i), str(j))
        elif left != None:
            dot3.edge(str(i), str(left))
        elif right != None:
            dot3.edge(str(i), str(right))
    dot3.render(directory='doctest-output')
    print('Время построения дерева решений равно：%f' % (t2 - t1))
    print('Время классификации тестовой выборки равно：%f' % (t3 - t2))
    print('Точность классификации：%f' % score)
    print('Параметр установленный на min_sample_leaf：%d' % min_sample_leaf)
    app = QApplication(sys.argv)
    ex = CART.Example()
    sys.exit(app.exec_())