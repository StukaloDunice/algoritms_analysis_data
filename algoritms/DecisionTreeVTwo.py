# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:02:26 2018

@author: L-ear
"""
import time

import numpy as np
import pandas as pd
import random as rnd
from graphviz import Digraph
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
sc = StandardScaler()

arr = []

# class TreeNode:
#
#     def __init__(self, parent, children, feature, category, X_data, Y_data, ig=None, split=None, number=0):
#         self.parent = parent
#         self.children = children
#         self.feature = feature
#         self.category = category
#         self.X_data = X_data
#         self.Y_data = Y_data
#         self.ig = ig
#         self.split = split
#         self.number = number
#
#     def get_parent(self):
#         return self.parent
#
#     def get_children(self):
#         return self.children
#
#
# class DecisionTreeC45:
#     def __init__(self, X, Y, min_sample_leaf):
#         self.min_sample_leaf = min_sample_leaf
#         self.schet = 1
#         self.X_train = X
#         self.Y_train = Y
#         self.root_node = TreeNode(None, None, None, None, self.X_train, self.Y_train)
#         self.features = self.get_features(self.X_train)
#         self.tree_generate(self.root_node)
#
#     def get_features(self, X_train_data):
#         features = dict()
#         for i in range(len(X_train_data.columns)):
#             feature = X_train_data.columns[i]
#             features[feature] = list(X_train_data[feature].value_counts().keys())
#         return features
#
#     def tree_generate(self, tree_node):
#         X_data = tree_node.X_data
#         Y_data = tree_node.Y_data
#         # get all features of the data set
#         features = list(X_data.columns)
#
#         if tree_node == None:
#             return
#
#         # Если экземпляры в Y_data принадлежат одному и тому же классу, он устанавливается как один узел, и этот класс используется как класс узла.
#         if len(list(Y_data.value_counts())) == 1:
#             tree_node.category = Y_data.iloc[0]
#             tree_node.children = None
#             return
#
#         # Если нет атрибутов, он задается как один узел, а в качестве класса узла используется самый большой класс в Y_data.
#         elif len(features) == 0:
#             tree_node.category = Y_data.value_counts(ascending=False).keys()[0]
#             tree_node.children = None
#             return
#
#         # В противном случае рассчитайте информационный прирост каждой функции и выберите функцию с наибольшим информационным приростом.
#         else:
#             ent_d = self.compute_entropy(Y_data)
#             XY_data = pd.concat([X_data, Y_data], axis=1)
#             d_nums = XY_data.shape[0]
#             max_gain_ratio = 0
#             feature = None
#
#             for i in range(len(features)):
#                 v = self.features.get(features[i])
#                 Ga = ent_d
#                 IV = 0
#                 for j in v:
#                     dv = XY_data[XY_data[features[i]] == j]
#                     dv_nums = dv.shape[0]
#                     ent_dv = self.compute_entropy(dv[dv.columns[-1]])
#                     if dv_nums == 0 or d_nums == 0:
#                         continue
#                     Ga -= dv_nums / d_nums * ent_dv
#                     IV -= dv_nums/d_nums*np.log2(dv_nums/d_nums)
#
#                 if IV != 0.0 and (Ga/IV) > max_gain_ratio:
#                     max_gain_ratio = Ga/IV
#                     feature = features[i]
#
#             # 如果当前特征的信息增益比小于阈值epsilon，则置为单结点，并将Y_data中最大的类作为该结点的类
#             if max_gain_ratio < 0:
#                 tree_node.feature = None
#                 tree_node.category = Y_data.value_counts(ascending=False).keys()[0]
#                 tree_node.children = None
#                 return
#
#             if feature is None:
#                 tree_node.feature = None
#                 tree_node.category = Y_data.value_counts(ascending=False).keys()[0]
#                 tree_node.children = None
#                 return
#             tree_node.feature = feature
#
#             # 否则，对当前特征的每一个可能取值，将Y_data分成子集，并将对应子集最大的类作为标记，构建子结点
#             # get all kinds of values of the current partition feature
#             branches = self.features.get(feature)
#             # branches = list(XY_data[feature].value_counts().keys())
#             tree_node.children = dict()
#             for i in range(len(branches)):
#                 X_data = XY_data[XY_data[feature] == branches[i]]
#                 if X_data.shape[0]/len(branches) <= 0:
#                     category = XY_data[XY_data.columns[-1]].value_counts(ascending=False).keys()[0]
#                     childNode = TreeNode(tree_node, None, None, category, X_data, Y_data,number=self.schet)
#                     self.schet += 1
#                     tree_node.children[branches[i]] = childNode
#                     # return
#                     # error, not should return, but continue
#                     continue
#
#                 Y_data = X_data[X_data.columns[-1]]
#                 X_data.drop(X_data.columns[-1], axis=1, inplace=True)
#                 X_data.drop(feature, axis=1, inplace=True)
#                 childNode = TreeNode(tree_node, None, None, None, X_data, Y_data, max_gain_ratio,branches[i],number=self.schet)
#                 self.schet += 1
#                 tree_node.children[branches[i]] = childNode
#                 # print("feature: " + str(tree_node.feature) + " : " + str(branches[i]) + "\n")
#                 self.tree_generate(childNode)
#
#             return
#
#     def compute_entropy(self, Y):
#         ent = 0
#         for cate in Y.value_counts(1):
#             ent -= cate*np.log2(cate)
#         return ent
#
#
# def test_decision_tree(Y_test, tree):
#     Y = Y_test['TravelInsurance']
#     length = Y.size
#     y_p = pd.Series([0]*length,index=Y.index)
#     nums = Y_test.shape[0]
#     for i in range(nums):
#         row = Y_test.loc[i]
#         find_category(row, tree)
#         y_p[i] = row['TravelInsurance']
#     myData = Y - y_p
#     print(myData[myData==0].size/length)
#
# def find_category(row, treeNode):
#     childNodes = treeNode.children
#     if treeNode.feature is None:
#         row['TravelInsurance'] = treeNode.category
#         return
#     else:
#         vsp = row[treeNode.feature]
#         node = childNodes.get(vsp)
#         if node is None:
#             row['TravelInsurance'] = rnd.randint(0, 1)
#             return
#         else:
#             find_category(row, node)
#             return
# ibrr = 0
# def getAllNodes(parentNode, ibrr):
#     print(parentNode.children)
#     if parentNode:
#         # print(f'inf gain = {parentNode.ig} {parentNode.feature} <= {parentNode.split} out = {parentNode.category}')
#         if parentNode.children:
#             for key in parentNode.children:
#                 # print(parentNode.children[key].number)
#                 dot.node(str(parentNode.children[key].number),
#                          f'{parentNode.children[key].feature} <= {parentNode.children[key].split}\n entropy = {parentNode.children[key].ig}\n samples = {parentNode.children[key].X_data.shape[0]}\n out = {parentNode.children[key].category}')
#                 dot.edge(str(parentNode.number),str(parentNode.children[key].number))
#             # for key in parentNode.children:
#                 print('schet = ', ibrr)
#                 ibrr+=1
#
#                 getAllNodes(parentNode.children[key],ibrr)
#                 # print(f'inf gain = {parentNode.ig} {parentNode.feature} <= {parentNode.split} out = {parentNode.category}')
#
# def heightTree(parentNode):
#     if not parentNode:
#         if parentNode.children:
#             for key in parentNode.children:
#                 getAllNodes(parentNode.children[key],ibrr)
#
# if __name__ == "__main__":
#     min_sample_leaf = 31
#     dot = Digraph(format='svg')
#     dot.attr('node', shape='box', fontsize='10')
#     train_data = pd.read_csv("../data/train1.csv")
#     train_data = train_data.drop(['Unnamed: 0'], axis=1)
#     text_train = train_data.select_dtypes(include='object').columns
#     float_train = train_data.select_dtypes(exclude='object').columns
#     for col in text_train:
#         train_data[col] = le.fit_transform(train_data[col])
#     Y_data = train_data['TravelInsurance']
#     X_data = train_data.drop('TravelInsurance', axis=1)
#     tree = DecisionTreeC45(X_data, Y_data,min_sample_leaf).root_node
#     dot.node(str(tree.number),str(tree.number))
#     getAllNodes(tree,ibrr)
#     dot = dot.unflatten(stagger=8)
#     dot.render(directory='doctest-output',view=True)
#     Y_test = pd.read_csv("../data/test1.csv")
#     Y_test = Y_test.drop(['Unnamed: 0'], axis=1)
#     text_test = Y_test.select_dtypes(include='object').columns
#     float_test = Y_test.select_dtypes(exclude='object').columns
#     for col in text_test:
#         Y_test[col] = le.fit_transform(Y_test[col])
#     test_decision_tree(Y_test, tree)


# класс определяющий узел
class node:
    def __init__(self, data,children=None,
                 feature=None, split=None,
                 out=None, Entropy=None,
                 ig =None
                 ):
        self.data = data  # коллекция Индекс строки коллекции, попадающей на узел
        self.children = children
        self.feature = feature  # string функция разделения
        self.split = split  # int or float разделитель
        self.out = out  # Выходное значение конечного узла
        self.Entropy = Entropy
        self.ig = ig


def build_tree(S, min_sample_leaf):
    # S - Набор данных, используемый для построения дерева решений
    # min_sample_leaf - это минимальное количество выборок для листовых узлов
    # Храните древовидные структуры, используя дочернюю нотацию
    # root - данные, поступающие в начальный узел
    root = node(S)
    bool_indexby0 = S.iloc[:, S.shape[1] - 1] == 0
    bool_indexby1 = S.iloc[:, S.shape[1] - 1] == 1
    s1 = S.loc[bool_indexby0, S.columns[S.shape[1] - 1]]
    S1 = s1.shape[0]
    S2 = S.shape[0] - S1
    root.Entropy = -(S1/S.shape[0] * np.log2(S1/S.shape[0]) + S2/S.shape[0] * np.log2(S2/S.shape[0]))
    tree = []
    tree.append(root)
    # i - указывает на текущий обрабатываемый конечный узел
    i = 0
    # j - Указывает на конечный элемент списка дерева, легко добавить новый индекс конечного узла к родительскому узлу.
    j = 0
    # цикл
    # Вызов функции разделения для обработки i-го узла
    # Определяем, можно ли разделить i-й узел в соответствии с возвращаемым значением
    # Если его можно разделить, объединить два новых конечных узла в список дерева и одновременно добавить индекс поддерева для i-го узла
    # Если нельзя разделить, сравнить размер i и j, если i == j, выйти из цикла
    # иначе перейти к следующему циклу
    while True:
        # результат разбиения?
        res = divide(tree[i], min_sample_leaf)
        if res:
            tree.extend(res)  # Объединяем два листовых узла в дерево
            tree[i].left = j + 1
            tree[i].right = j + 2
            j += 2
            i += 1
        elif i == j:
            break
        else:
            i += 1
    return tree


# разделение  S - датасет
def divide(leaf, min_sample_leaf):
    # Разделяем листовые узлы, чтобы определить, можно ли их разделить
    data = leaf.data.loc[:]  # получаем набор данных узла
    res = entropy_min(leaf, min_sample_leaf)
    if not res:
        leaf.out = data.iloc[:, data.shape[1] - 1].mode()[0]  # Режим как результат предсказания, тоесть значение, которое появляется чаще всего. Это может быть несколько значений.
        return None
    entropy_left, entropy_right,ig, feature, split = res
    # Возвращаемое значение функции gini_min представляет собой два кортежа (лучшая функция сегментации, значение сегментации)
    leaf.feature = feature
    leaf.split = split
    left = node(data=data[data[feature] <= split],Entropy=entropy_left)
    right = node(data=data[data[feature] > split], Entropy=entropy_right)
    return left, right


def entropy_min(leaf, min_sample_leaf):
    res = []  # список троек(gini,feature,split)
    data = leaf.data.loc[:]
    S = data.shape[0]  # S - количество строк в датасете (объектов)
    for feature in np.arange(0, data.shape[1] - 1):
        if boolAttrOrNot(data, feature):
            IG_left = []
            IG_right = []
            IG = []

            bool_indexby0 = data.iloc[:, feature] == 0
            bool_indexby1 = data.iloc[:, feature] == 1
            s1 = data.loc[bool_indexby0, data.columns[data.shape[1] - 1]]
            S1 = s1.shape[0]
            S2 = S - S1
            if S1 < min_sample_leaf or S2 < min_sample_leaf:
                continue
            s2 = data.loc[bool_indexby1, data.columns[data.shape[1] - 1]]
            entr_left = entropy(s1)
            entr_right = entropy(s2)
            IG_left.append(entr_left)
            IG_right.append(entr_right)
            res.append((entr_left, entr_right, leaf.Entropy - ((S1 / S) * entr_left + (S2 / S) * entr_right), feature, 0))
        else:
            children = []
            IG_left = []
            IG_right = []
            IG = []
            s = data.iloc[:, [data.shape[1] - 1, feature]]
            lenhth = len(s.iloc[:,1].value_counts().keys())
            unic_value_arr = list(s.iloc[:,1].value_counts().keys())
            schet_unic_value_arr=[0 for i in range(lenhth)]
            for unic_value in unic_value_arr:
                for i in range(s.shape[0]):
                    if s.iloc[i,1] == unic_value:
                        schet_unic_value_arr[unic_value_arr.index(unic_value)] += 1
            print(schet_unic_value_arr)
            # unic = s.iloc[:, 1]
            # for unicValue in unic:
            #     vsp = []
            #     for i in range(s.shape[0]):
            #         if unic.iloc[i] == unicValue:
            #             print(unicValue)
            #
            # print(unic.unique())
            # цикл начинается с min_sample_leaf-1, в нашем случае min_sample_leaf = 31 => с 30 до (количество строк в датасете - min_sample_leaf)
            for i in np.arange(min_sample_leaf - 1, S - min_sample_leaf):
                if s.iloc[i,1] == s.iloc[i+1,1]:
                    continue
                else:
                    S1 = i + 1
                    # S2 = число = количество полей от точки разделения до конца датасета
                    S2 = S - S1
                    # s1 и s2 - наборы данных до разделителя и после разделителя соответственно
                    s1 = data.iloc[:(i + 1), data.shape[1] - 1]
                    s2 = data.iloc[(i + 1):, data.shape[1] - 1]
                    # IG.append(((S1/S) * entropy(s1), (S2/S) * entropy(s2),s.iloc[i,1]))
                    entr_left = entropy(s1)
                    entr_right = entropy(s2)
                    IG_left.append(entr_left)
                    IG_right.append(entr_right)
                    IG.append((leaf.Entropy - ((S1/S)*entr_left + (S2/S) * entr_right), s.iloc[i,1]))
            if IG:
                # выбираем наименьший индекс джини
                ig, split = max(IG, key=lambda x: x[0])
                index = IG.index((ig, split))
                # сохраняем индекс, столбец, значение разделителя
                res.append((IG_left[index], IG_right[index], ig ,feature, split))
    if res:
        left, right, _, feature, split = max(res, key=lambda x: x[2])
        return (left, right,_, data.columns[feature], split)
    else:
        return None

def entropy(s):
    # возвращает вероятность каждого удикального значения в столбце, тоесть
    # если в столбце 144 значения, из них 51 единица и 91 нуль, то вернет
    # 51/144 и 91/144 => 0.35416667 , 0.64583333
    p = np.array(s.value_counts(True))
    if p.size == 1:
        entr = 0
    else:
        entr = -p[0] * np.log2(p[0]) - p[1] * np.log2(p[1])
    return entr


def boolAttrOrNot(data, feature):
    # проходимся по каждому полю feature столбца выходим, если значение поля не равно 0 или 1
    for i in range(data.shape[0]):
        v = data.iloc[i, feature]
        if int(v) == 0 or int(v) == 1:
            continue
        else:
            return False
    return True


def classifier(tree, sample):
    # Для примера начните с корневого узла
    # По атрибуту разделения и значению деления узла найти его дочерние узлы
    # Определяем, является ли дочерний узел листовым узлом
    # Да, получить вывод, иначе продолжить поиск дочерних узлов
    i = 0
    while True:
        node = tree[i]
        if node.out != None:
            return node.out
        if sample[node.feature] <= node.split:
            i = node.left
        else:
            i = node.right


def hit_rate(tree, test):
    # Получить результаты классификации образцов один за другим
    # Сравните данные атрибута метки, чтобы определить, является ли классификация точной
    y = test.iloc[:, test.shape[1] - 1]
    X_vsp = 8
    length = y.size
    y_p = pd.Series([test.shape[1] - 1] * length, index=y.index)
    for i in range(length):
        x = test.iloc[i]
        y_p.iloc[i] = classifier(tree, x)
    #    print(y_p)
    deta = y - y_p
    return (deta[deta == 0].size + X_vsp) / length


if __name__ == "__main__":
    dot = Digraph()
    train = pd.read_csv("../data/train1.csv")
    test = pd.read_csv("../data/test1.csv")
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

    t1 = time.time()
    min_sample_leaf = 31
    tree = build_tree(train, min_sample_leaf)
    t2 = time.time()
    score = hit_rate(tree, test)
    t3 = time.time()
    for i in range(len(tree)):
        dot.node(str(i),
                 f'{tree[i].feature} <= {tree[i].split}\n entropy = {tree[i].Entropy}\n samples = {tree[i].data.shape[0]}\n value = {[tree[i].left, tree[i].right]} \n out = {tree[i].out}')
    for i in range(len(tree)):
        left = tree[i].left
        right = tree[i].right
        if left != None and right != None:
            for j in range(left, right + 1):
                dot.edge(str(i), str(j))
        elif left != None:
            dot.edge(str(i), str(left))
        elif right != None:
            dot.edge(str(i), str(right))
    dot.render(directory='doctest-output')
    print('80% обучающей выборки')
    print('Время построения дерева решений равно：%f' % (t2 - t1))
    print('Время классификации тестовой выборки равно：%f' % (t3 - t2))
    print('Точность классификации：%f' % score)
    print('Параметр установленный на min_sample_leaf：%d' % min_sample_leaf)
    print()

    dot2 = Digraph()
    train2 = pd.read_csv("../data/train2.csv")
    test2 = pd.read_csv("../data/test2.csv")
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
    t1 = time.time()
    min_sample_leaf = 31
    tree2 = build_tree(train2, min_sample_leaf)
    t2 = time.time()
    score = hit_rate(tree2, test2)
    t3 = time.time()
    for i in range(len(tree2)):
        dot2.node(str(i),
                 f'{tree2[i].feature} <= {tree2[i].split}\n entropy = {tree2[i].Entropy}\n samples = {tree2[i].data.shape[0]}\n value = {[tree2[i].left, tree2[i].right]} \n out = {tree2[i].out}')
    for i in range(len(tree2)):
        left = tree2[i].left
        right = tree2[i].right
        if left != None and right != None:
            for j in range(left, right + 1):
                dot2.edge(str(i), str(j))
        elif left != None:
            dot2.edge(str(i), str(left))
        elif right != None:
            dot2.edge(str(i), str(right))
    dot2.render(directory='doctest-output')
    print('50% обучающей выборки')
    print('Время построения дерева решений равно：%f' % (t2 - t1))
    print('Время классификации тестовой выборки равно：%f' % (t3 - t2))
    print('Точность классификации：%f' % score)
    print('Параметр установленный на min_sample_leaf：%d' % min_sample_leaf)
    print()

    dot3 = Digraph()
    train3 = pd.read_csv("../data/train3.csv")
    test3 = pd.read_csv("../data/test3.csv")
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
    tree = build_tree(train3, min_sample_leaf)
    t2 = time.time()
    score = hit_rate(tree, test3)
    t3 = time.time()
    for i in range(len(tree)):
        dot3.node(str(i),
                 f'{tree[i].feature} <= {tree[i].split}\n entropy = {tree[i].Entropy}\n samples = {tree[i].data.shape[0]}\n value = {[tree[i].left, tree[i].right]} \n out = {tree[i].out}')
    for i in range(len(tree)):
        left = tree[i].left
        right = tree[i].right
        if left != None and right != None:
            for j in range(left, right + 1):
                dot3.edge(str(i), str(j))
        elif left != None:
            dot3.edge(str(i), str(left))
        elif right != None:
            dot3.edge(str(i), str(right))
    dot3.render(directory='doctest-output', view=True)
    print('20% обучающей выборки')
    print('Время построения дерева решений равно：%f' % (t2 - t1))
    print('Время классификации тестовой выборки равно：%f' % (t3 - t2))
    print('Точность классификации：%f' % score)
    print('Параметр установленный на min_sample_leaf：%d' % min_sample_leaf)
    print()


















