import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from algoritms import DecisionTree as CART
from algoritms import DecisionTreeVTwo as C4_5
import time
import matplotlib.pyplot as plt

le = LabelEncoder()
sc = StandardScaler()

if __name__ == "__main__":
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
    C4_5_scores = []
    CART_scores = []
    for i in range(100):
        t1 = time.time()
        tree = CART.build_tree(train ,i+1)
        t2 = time.time()
        score_CART = CART.hit_rate(tree, test)
        t3 = time.time()
        tree = CART.build_tree(train, i + 1)
        t4 = time.time()
        score_C4_5 = CART.hit_rate(tree, test)
        t5 = time.time()
        CART_scores.append((score_CART, i+1, t2 - t1, t3 - t2))
        C4_5_scores.append((score_C4_5, i + 1, t4 - t3, t5 - t4))

    score = []
    min_sample_leaf = []
    time_created = []
    time_classifire = []
    for i in C4_5_scores:
        score.append(i[0])
        min_sample_leaf.append(i[1])
        time_created.append(i[2])
        time_classifire.append(i[3])
    best = max(C4_5_scores, key=lambda x: x[0])
    low = min(C4_5_scores, key=lambda x: x[0])

    plt.subplots_adjust(hspace=0.5)

    plt.figure(figsize=(15, 20))
    plt.subplot(3, 3, 1)
    plt.plot(min_sample_leaf, score)
    plt.scatter(best[1], best[0], c='red')
    plt.hlines(best[0], 0, best[1], colors='green')
    plt.vlines(best[1], low[0], best[0], colors='green')
    plt.legend(title='min_sample_leaf = ' + str(best[1]))
    plt.title("min_sample_leaf")
    plt.subplot(3, 3, 2)
    plt.plot(min_sample_leaf, time_created)
    plt.legend(title=str(best[2]))
    plt.title(f'Decision tree C4.5 80% train data\n\ntime_created')
    plt.subplot(3, 3, 3)
    plt.plot(min_sample_leaf, time_classifire)
    plt.title("time_classifire")


    score = []
    min_sample_leaf = []
    time_created = []
    time_classifire = []
    for i in CART_scores:
        score.append(i[0])
        min_sample_leaf.append(i[1])
        time_created.append(i[2])
        time_classifire.append(i[3])
    best = max(CART_scores, key=lambda x: x[0])
    low = min(CART_scores, key=lambda x: x[0])
    plt.figure(figsize=(15, 20))
    plt.subplot(3, 3, 4)
    plt.plot(min_sample_leaf, score)
    plt.scatter(best[1], best[0], c='red')
    plt.hlines(best[0], 0, best[1], colors='green')
    plt.vlines(best[1], low[0], best[0], colors='green')
    plt.legend(title='min_sample_leaf = ' + str(best[1]))
    plt.title("min_sample_leaf")
    plt.subplot(3, 3, 5)
    plt.plot(min_sample_leaf, time_created)
    plt.legend(title=str(best[2]))
    plt.title(f'Decision tree CART 80% train data\n\ntime_created')
    plt.subplot(3, 3, 6)
    plt.plot(min_sample_leaf, time_classifire)
    plt.title("time_classifire")

    plt.show()