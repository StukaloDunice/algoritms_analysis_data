from algoritms import DecisionTreeCART as DT
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
sc = StandardScaler()

def RandomForest(train, n_trees, min_sample_leaf, ip, jp):
    forest = []# Сохраняем все деревья решений, полученные в результате обучения
    # количество столбцов для обучения n-го дерева
    fn = int(jp*(train.shape[1]-1))
    for n in range(n_trees):
        t1 = time.time()
        # выбирааем fn(5) рандомных столбцов
        sf = np.random.choice(
                np.arange(0,train.shape[1]-1),
                fn,
                replace=False)
        sf = np.append(train.shape[1]-1,sf) # Убедитесь, что метка находится в первом столбце
        # разворачиваем список, чтобы целевая переменная стояла в первом столбце
        sf = np.flip(sf)
        # берем все строки для тестовой выборки
        train_n = train.iloc[:,sf]
        # берем часть выборки применив найденный коэффициент
        p = np.random.random_sample()*(1-ip)+ip
        train_n = train_n.loc[
                np.random.choice(train_n.index,
                                 int(p*train_n.index.size),
                                 replace=False)]
        # train_n — обучающая выборка случайно выбранного n-го дерева.
        # обучаем дерево решений и сохраняем в массив "лес"
        forest.append(DT.build_tree(train_n, min_sample_leaf))
        t2 = time.time()
        # print('Время построения %d модели дерева = %f'%(n,t2-t1))
    return forest   



def hit_rate(forest, test):
    # Получить данные атрибутов, не являющихся метками, в тестовом наборе
    # Получить результаты классификации образцов один за другим
    # Сравните данные атрибута метки, чтобы определить, является ли классификация точной
    y = test.iloc[:,test.shape[1]-1]
    length = y.size
    y_p = pd.Series([test.shape[1]-1]*length,index=y.index)
    n_trees = len(forest)
    res = [0]*n_trees # Сохранение результатов прогнозирования для каждого дерева
    for i in range(length):
        x = test.iloc[i]
        for t in range(n_trees):
            res[t] = DT.classifier(forest[t],x)
        y_p.iloc[i] = max(res,key=res.count)
    deta = y-y_p
    return deta[deta==0].size/length
    
if __name__ == "__main__":
    ip = 0.85
    jp = 0.7
    n_trees = 30
    min_sample_leaf = 1
    # # Вышеупомянутый параметр должен быть скорректирован
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
    forest_scores = []
    for i in range(100):
        t1 = time.time()
        forest = RandomForest(train,i+1,min_sample_leaf,ip,jp)
        t2 = time.time()
        score = hit_rate(forest,test)
        t3 = time.time()
        forest_scores.append((score, i + 1, t2 - t1, t3 - t2))
    print(forest_scores)
    # print('Время классификации набора тестов равно %f'%(t2-t1))
    # print('Значения параметров: n_trees=%d,min_sample_leaf=%d,ip=%f,jp=%f'%(n_trees,min_sample_leaf,ip,jp))
    # print('Точность алгоритма = %f'%score)

    train2 = pd.read_csv("../data/train2.csv")
    test2 = pd.read_csv("../data/test2.csv")
    train2 = train2.drop(['Unnamed: 0'], axis=1)
    test2 = test2.drop(['Unnamed: 0'], axis=1)
    text_train2 = train2.select_dtypes(include='object').columns
    float1_train2 = train2.select_dtypes(exclude='object').columns
    text_test2 = test2.select_dtypes(include='object').columns
    float1_test3 = test2.select_dtypes(exclude='object').columns
    for col in text_train2:
        train2[col] = le.fit_transform(train2[col])
    for col in text_test2:
        test2[col] = le.fit_transform(test2[col])
    forest_scores = []
    for i in range(100):
        t1 = time.time()
        forest = RandomForest(train2, i+1,min_sample_leaf, ip, jp)
        t2 = time.time()
        score = hit_rate(forest, test2)
        t3 = time.time()
        forest_scores.append((score, i + 1, t2 - t1, t3 - t2))
    print(forest_scores)
    # print('Время классификации набора тестов равно %f'%(t2-t1))
    # print('Значения параметров: n_trees=%d,min_sample_leaf=%d,ip=%f,jp=%f'%(n_trees,min_sample_leaf,ip,jp))
    # print('Точность алгоритма = %f'%score)


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
    forest_scores = []
    for i in range(100):
        t1 = time.time()
        forest = RandomForest(train3, i+1,min_sample_leaf, ip, jp)
        t2 = time.time()
        score = hit_rate(forest, test3)
        t3 = time.time()
        forest_scores.append((score, i + 1, t2 - t1, t3 - t2))
    print(forest_scores)
    # print('Время классификации набора тестов равно %f'%(t2-t1))
    # print('Значения параметров: n_trees=%d,min_sample_leaf=%d,ip=%f,jp=%f'%(n_trees,min_sample_leaf,ip,jp))
    # print('Точность алгоритма = %f'%score)
