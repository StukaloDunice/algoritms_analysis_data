import math
import time

import numpy as np
import pandas as pd
import random as rnd
from graphviz import Digraph
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
sc = StandardScaler()

arr = []

class node:
    def __init__(self, data,
                 left=None, right=None,
                 feature=None, split=None,
                 out=None, Entropy=None,
                 ig =None
                 ):
        self.data = data  # коллекция Индекс строки коллекции, попадающей на узел
        self.left = left  # int индекс левого поддерева
        self.right = right  # int индекс правого поддерева
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
    res = ig_ratio_min(leaf, min_sample_leaf)
    if not res:
        leaf.out = data.iloc[:, data.shape[1] - 1].mode()[0]  # Режим как результат предсказания, тоесть значение, которое появляется чаще всего. Это может быть несколько значений.
        return None
    feature, split = res
    # Возвращаемое значение функции gini_min представляет собой два кортежа (лучшая функция сегментации, значение сегментации)
    leaf.feature = feature
    leaf.split = split
    left = node(data=data[data[feature] <= split])
    right = node(data=data[data[feature] > split])
    return left, right

def ig_ratio_min(leaf, min_sample_leaf):
    data_leaf = leaf.data
    number_row_data_leaf = data_leaf.shape[0]
    number_col_data_leaf = data_leaf.shape[1]
    leaf.Entropy = entropy_node(data_leaf)
    all_split = []
    for attribute in np.arange(0, number_col_data_leaf -1):
        target_and_attribute = data_leaf.iloc[:, [attribute, number_col_data_leaf - 1]]
        if boolAttrOrNot(target_and_attribute):
            ig = leaf.Entropy - entropy_split(target_and_attribute)
            bool_indexby0 = data_leaf.iloc[:,attribute] == 0
            bool_indexby1 = data_leaf.iloc[:, attribute] == 1
            s1 = data_leaf.loc[bool_indexby0, data_leaf.columns[attribute]]
            S1 = s1.shape[0]
            S2= number_row_data_leaf - S1
            if S1 <= min_sample_leaf or S2 <= min_sample_leaf:
                continue
            p_null =S1/number_row_data_leaf
            p_one = S2/number_row_data_leaf
            ig_ratio = ig / -(p_null*np.log2(p_null) + p_one*np.log2(p_one) )
            all_split.append((ig_ratio, target_and_attribute.columns[0], 0))
        else:
            ig_list = []
            target_and_attribute = target_and_attribute.sort_values(target_and_attribute.columns[0])
            for i in np.arange(min_sample_leaf -1, number_row_data_leaf - min_sample_leaf):
                if target_and_attribute.iloc[i,0] == target_and_attribute.iloc[i+1,0]:
                    continue
                number_split_attribute_do = target_and_attribute.iloc[:, 0] == target_and_attribute.iloc[i,0]
                number_split_attribute_posle = target_and_attribute.iloc[:, 0] != target_and_attribute.iloc[i,0]
                number_split_attribute_do = target_and_attribute.loc[number_split_attribute_do]
                number_split_attribute_posle = target_and_attribute.loc[
                    number_split_attribute_posle]
                S1 = number_split_attribute_do.shape[0]
                S2 = number_row_data_leaf - S1
                entr = (S1/number_row_data_leaf) * entropy_node(number_split_attribute_do) + (S2/number_row_data_leaf) * entropy_node(number_split_attribute_posle)
                ig = leaf.Entropy - entr
                p_S1 = S1/number_row_data_leaf
                p_S2 = S2/number_row_data_leaf
                ig_ratio = ig/-(p_S1 * np.log2(p_S1) + p_S2 * np.log2(p_S2))
                ig_list.append((ig_ratio, target_and_attribute.columns[0], target_and_attribute.iloc[i,0]))
            if ig_list:
                ig_ratio, attribute, split = max(ig_list,key=lambda x:x[0])
                all_split.append((ig_ratio,attribute, split))

    if all_split:
        ig_ratio, argument, split = max(all_split, key=lambda x:x[0])
        return (argument, split)
    else:
        return None

def entropy_node(data):
    number_row_input_data = data.shape[0]
    number_col_input_data = data.shape[1]
    # нули в таргет столбце
    null_in_col = data.iloc[:, number_col_input_data - 1] == 0
    null_in_col = data.loc[null_in_col, data.columns[number_col_input_data - 1]]
    number_null_in_col = null_in_col.shape[0]
    number_one_in_col = number_row_input_data - number_null_in_col
    if number_one_in_col == 0 or number_null_in_col == 0:
        return 0
    p_null = number_null_in_col / number_row_input_data
    p_one = number_one_in_col / number_row_input_data
    return -(p_null * np.log2(p_null) + p_one * np.log2(p_one))

def entropy_split(two_cols):
    size_input_data = two_cols.shape[0]
    null_in_col_attribute = two_cols.iloc[:, 0] == 0
    null_in_col_attribute = two_cols.loc[null_in_col_attribute]
    number_null_attribute = null_in_col_attribute.shape[0]
    one_in_col_attribute = two_cols.iloc[:, 0] == 1
    one_in_col_attribute = two_cols.loc[one_in_col_attribute]
    number_one_attribute = one_in_col_attribute.shape[0]
    p_null_attribute = number_null_attribute/size_input_data
    p_one_attribute = number_one_attribute/size_input_data
    if p_one_attribute ==0 or p_one_attribute == 0:
        res_entropy = 0
    else:
        res_entropy = p_null_attribute*entropy_node(null_in_col_attribute) + p_one_attribute*entropy_node(one_in_col_attribute)
    # print(res_entropy)
    return res_entropy

def boolAttrOrNot(data):
    # проходимся по каждому полю feature столбца выходим, если значение поля не равно 0 или 1
    for i in range(data.shape[0]):
        v = data.iloc[i, 0]
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
    return deta[deta == 0].size / length


if __name__ == "__main__":
    min_sample_leaf = 31
    dot = Digraph(format='png')
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
    trees_scores = []
    for i in range(100):
        t1 = time.time()
        tree = build_tree(train, i + 1)
        t2 = time.time()
        score = hit_rate(tree, test)
        t3 = time.time()
        trees_scores.append((score, i + 1, t2 - t1, t3 - t2))
    print(trees_scores)
    # for i in range(len(tree)):
    #     dot.node(str(i),
    #              f'{tree[i].feature} <= {tree[i].split}\n gini = {tree[i].Gini}\n samples = {tree[i].data_index.size}\n value = {[tree[i].left, tree[i].right]} \n out = {tree[i].out}')
    # for i in range(len(tree)):
    #     left = tree[i].get_left()
    #     right = tree[i].get_right()
    #     if left != None and right != None:
    #         for j in range(left, right + 1):
    #             dot.edge(str(i),str(j))
    #     elif left != None:
    #         dot.edge(str(i), str(left))
    #     elif right != None:
    #         dot.edge(str(i),str(right))
    # dot.render(directory='doctest-output',view=True)
    # print('Время построения дерева решений равно：%f'%(t2-t1))
    # print('Время классификации тестовой выборки равно：%f'%(t3-t2))
    # print('Точность классификации：%f'%score)
    # print('Параметр установленный на min_sample_leaf：%d'%min_sample_leaf)

    dot2 = Digraph(format='png')
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
    trees_scores = []
    for i in range(100):
        t1 = time.time()
        tree = build_tree(train2, i + 1)
        t2 = time.time()
        score = hit_rate(tree, test2)
        t3 = time.time()
        trees_scores.append((score, i + 1, t2 - t1, t3 - t2))
    print(trees_scores)
    # for i in range(len(tree)):
    #     dot2.node(str(i),
    #              f'{tree[i].feature} <= {tree[i].split}\n gini = {tree[i].Gini}\n samples = {tree[i].data_index.size}\n value = {[tree[i].left, tree[i].right]} \n out = {tree[i].out}')
    # for i in range(len(tree)):
    #     left = tree[i].get_left()
    #     right = tree[i].get_right()
    #     if left != None and right != None:
    #         for j in range(left, right + 1):
    #             dot2.edge(str(i), str(j))
    #     elif left != None:
    #         dot2.edge(str(i), str(left))
    #     elif right != None:
    #         dot2.edge(str(i), str(right))
    # dot2.render(directory='doctest-output')
    # print('Время построения дерева решений равно：%f' % (t2 - t1))
    # print('Время классификации тестовой выборки равно：%f' % (t3 - t2))
    # print('Точность классификации：%f' % score)
    # print('Параметр установленный на min_sample_leaf：%d' % min_sample_leaf)

    # dot3 = Digraph(format='png')
    # train3 = pd.read_csv("../data/train3.csv")
    # test3 = pd.read_csv("../data/test3.csv")
    # train3 = train3.drop(['Unnamed: 0'], axis=1)
    # test3 = test3.drop(['Unnamed: 0'], axis=1)
    # text_train3 = train3.select_dtypes(include='object').columns
    # float1_train3 = train3.select_dtypes(exclude='object').columns
    # text_test3 = test3.select_dtypes(include='object').columns
    # float1_test3 = test3.select_dtypes(exclude='object').columns
    # for col in text_train3:
    #     train3[col] = le.fit_transform(train3[col])
    # for col in text_test3:
    #     test3[col] = le.fit_transform(test3[col])
    # trees_scores = []
    # for i in range(100):
    #     t1 = time.time()
    #     min_sample_leaf = 31
    #     tree = build_tree(train3, i + 1)
    #     t2 = time.time()
    #     score = hit_rate(tree, test3)
    #     t3 = time.time()
    #     trees_scores.append((score, i + 1, t2 - t1, t3 - t2))
    # print(trees_scores)
    # # for i in range(len(tree)):
    # #     dot3.node(str(i),
    # #               f'{tree[i].feature} <= {tree[i].split}\n gini = {tree[i].Gini}\n samples = {tree[i].data_index.size}\n value = {[tree[i].left, tree[i].right]} \n out = {tree[i].out}')
    # # for i in range(len(tree)):
    # #     left = tree[i].get_left()
    # #     right = tree[i].get_right()
    # #     if left != None and right != None:
    # #         for j in range(left, right + 1):
    # #             dot3.edge(str(i), str(j))
    # #     elif left != None:
    # #         dot3.edge(str(i), str(left))
    # #     elif right != None:
    # #         dot3.edge(str(i), str(right))
    # # dot3.render(directory='doctest-output')
    # # print('Время построения дерева решений равно：%f' % (t2 - t1))
    # # print('Время классификации тестовой выборки равно：%f' % (t3 - t2))
    # # print('Точность классификации：%f' % score)
    # # print('Параметр установленный на min_sample_leaf：%d' % min_sample_leaf)


















