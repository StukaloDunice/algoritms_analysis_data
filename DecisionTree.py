# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:02:26 2018

@author: L-ear
"""
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
sc = StandardScaler()
# класс определяющий узел
class node:
    def __init__(self,data_index,
                 left=None,right=None,
                 feature=None,split=None,
                 out = None,Gini=None
                 ):
        self.data_index = data_index# коллекция Индекс строки коллекции, попадающей на узел
        self.left = left# int индекс левого поддерева
        self.right = right# int индекс правого поддерева
        self.feature = feature# string функция разделения
        self.split = split# int or float разделитель
        self.out = out# Выходное значение конечного узла
        self.Gini = Gini
    def get(self):
        return([self.data_index, self.left, self.right, self.feature, self.split, self.out, self.Gini])

def build_tree(S, min_sample_leaf):
    # S - Набор данных, используемый для построения дерева решений
    # min_sample_leaf - это минимальное количество выборок для листовых узлов
    # Храните древовидные структуры, используя дочернюю нотацию
    # root - данные, поступающие в начальный узел
    root = node(S.index)
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
        res = divide(S, tree[i], min_sample_leaf)
        if res:
            tree.extend(res)# Объединяем два листовых узла в дерево
            tree[i].left = j+1
            tree[i].right = j+2
            j += 2
            i += 1
        elif i == j:
            break
        else:
            i += 1
    return tree

# разделение  S - датасет
def divide(S, leaf, min_sample_leaf):
    # Разделяем листовые узлы, чтобы определить, можно ли их разделить
    data = S.loc[leaf.data_index]# получаем набор данных узла
    res = gini_min(data,min_sample_leaf)
    if not res:
        leaf.out = data.iloc[:,data.shape[1]-1].mode()[0] # Режим как результат предсказания, тоесть значение, которое появляется чаще всего. Это может быть несколько значений.
        return None
    Gini, feature, split = res
    # Возвращаемое значение функции gini_min представляет собой два кортежа (лучшая функция сегментации, значение сегментации)
    leaf.feature = feature
    leaf.split = split
    leaf.Gini = Gini
    left = node(data[data[feature] <= split].index)
    right = node(data[data[feature] > split].index)
    return left, right


def gini_min(data, min_sample_leaf):
    # Получение лучшего разделения в наборе данных в соответствии с коэффициентом Джини
    res = [] # список троек(gini,feature,split)
    S = data.shape[0] # S - количество столбцов датасете (особенности или свойства)
    for feature in np.arange(0,data.shape[1]-1):
        # Сначала определите, является ли столбец переменной onehot, чтобы избежать сортировки переменной onehot.
        if is_one_hot(data,feature):
            bool_indexby0 = data.iloc[:,feature] == 0
            s1 = data.loc[bool_indexby0,data.columns[data.shape[1]-1]]
            S1 = s1.shape[0]
            S2 = S-S1
            if S1<min_sample_leaf or S2<min_sample_leaf:
                continue
            s2 = data.loc[not bool_indexby0,data.columns[data.shape[1]-1]]
            res.append(((S1*gini(s1) + S2*gini(s2))/S,feature,0))
        else:
            Gini_list = []# Список из двух кортежей (gini,split), в котором хранится оптимальное значение Джини и точка разделения каждой функции.
            # iloc помогает выбрать ячейки датасета, в нашем случае мы берем все строки и столбцы от 0 до feature
            # целева переменная на 0 месте
            s = data.iloc[:,[data.shape[1]-1,feature]]
            # сортируем первые два столбца по первому столбцу по возрастанию
            s = s.sort_values(s.columns[1])
            # цикл начинается с min_sample_leaf-1, в нашем случае min_sample_leaf = 31 => с 30 до (количество столбцов в датасете - min_sample_leaf)
            for i in np.arange(min_sample_leaf-1,S-min_sample_leaf):
                # ищем переход значения в поле столбца, тоесть с 0 на 1, с 1 на 2 и тп
                # ищем не по 0 столбцу, потому что в нулевом целевая переменная
                if s.iloc[i,1] == s.iloc[i+1,1]:
                    continue
                else:
                    # S1 = число = количество полей от начала датасета до точки разделения
                    # для примера, если датасет состоит из 600 строк
                    # и точка разделения (перехода с 0 на 1) на 143 позиции
                    # то S1 = 144, а S2 = 456
                    S1 = i+1
                    # S2 = число = количество полей от точки разделения до конца датасета
                    S2 = S-S1
                    # s1 и s2 - наборы данных до разделителя и после разделителя соответственно
                    s1 = data.iloc[:(i+1),data.shape[1]-1]
                    s2 = data.iloc[(i+1):,data.shape[1]-1]
                    # добавляем в список (Gini_list) наш найденный индекс джини
                    # для разбиения в узле и значение поля по которому было разбиение
                    Gini_list.append(((S1*gini(s1) + S2*gini(s2))/S,s.iloc[i,1]))
            # бывают случаи, Gini_list пуст
            if Gini_list:
                # выбираем наименьший индекс джини
                Gini_min,split = min(Gini_list,key=lambda x:x[0])
                # сохраняем индекс, столбец, значение разделителя
                res.append((Gini_min,feature,split))
    # res также может быть пустым
    if res:
        _,feature,split = min(res,key=lambda x:x[0])
        print(_)
        return (_,data.columns[feature],split)
    else:
        return None
    
       
def gini(s):
    # возвращает вероятность каждого удикального значения в столбце, тоесть
    # если в столбце 144 значения, из них 51 единица и 91 нуль, то вернет
    # 51/144 и 91/144 => 0.35416667 , 0.64583333
    p = np.array(s.value_counts(True))
    # np.square([0.64583,0.35416667]) - возводит в квадрат каждый элемент
    # вернет [0.41709639, 0.12543403]
    # np.sum - суммирует эти два элемента
    # 1-sum(pi^2)
    return 1-np.sum(np.square(p))


def is_one_hot(data,feature):
    # проходимся по каждому полю feature столбца выходим, если значение поля не равно 0 или 1
    for i in range(data.shape[0]):
        v = data.iloc[i,feature]
        if v != 0 or v != 1:
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
    y = test.pop(test.columns[test.shape[1]-1])
    length = y.size
    y_p = pd.Series([test.shape[1]-1]*length,index=y.index)
    for i in range(length):
        x = test.iloc[i]
        y_p.iloc[i] = classifier(tree,x)
#    print(y_p)
    deta = y-y_p
    return deta[deta==0].size/length
    
if __name__ == "__main__":
    train = pd.read_csv("data/train1.csv")
    test = pd.read_csv("data/test1.csv")
    # df = pd.DataFrame(data)
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
    # X = df.drop(['TravelInsurance'], axis=1)
    # y = df.TravelInsurance

    t1 = time.time()
    min_sample_leaf = 31
    tree = build_tree(train ,min_sample_leaf)
    t2 = time.time()
    score = hit_rate(tree, test)
    t3 = time.time()
    for i in tree:
        print(i.get())
    print('Время построения дерева решений равно：%f'%(t2-t1))
    print('Время классификации тестовой выборки равно：%f'%(t3-t2))
    print('Точность классификации：%f'%score)
    print('Параметр установленный на min_sample_leaf：%d'%min_sample_leaf)
    










