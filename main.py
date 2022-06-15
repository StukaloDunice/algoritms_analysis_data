from PyQt5.QtWidgets import QApplication
import sys

from gui import MainWindow



if __name__ == "__main__":
    # dot = Digraph()
    #
    # dot2 = Digraph()
    #
    # dot3 = Digraph(format='png')
    #
    # t1 = time.time()
    # min_sample_leaf = 31
    # tree = CART.build_tree(train ,min_sample_leaf)
    # t2 = time.time()
    # score = CART.hit_rate(tree, test)
    # t3 = time.time()
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
    # dot.render(directory='doctest-output')
    # print('Время построения дерева решений равно：%f'%(t2-t1))
    # print('Время классификации тестовой выборки равно：%f'%(t3-t2))
    # print('Точность классификации：%f'%score)
    # print('Параметр установленный на min_sample_leaf：%d'%min_sample_leaf)
    #
    #
    # t1 = time.time()
    # min_sample_leaf = 31
    # tree = CART.build_tree(train2, min_sample_leaf)
    # t2 = time.time()
    # score = CART.hit_rate(tree, test2)
    # t3 = time.time()
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
    #
    #
    # t1 = time.time()
    # min_sample_leaf = 31
    # tree = CART.build_tree(train3, min_sample_leaf)
    # t2 = time.time()
    # score = CART.hit_rate(tree, test3)
    # t3 = time.time()
    # for i in range(len(tree)):
    #     dot3.node(str(i),
    #               f'{tree[i].feature} <= {tree[i].split}\n gini = {tree[i].Gini}\n samples = {tree[i].data_index.size}\n value = {[tree[i].left, tree[i].right]} \n out = {tree[i].out}')
    # for i in range(len(tree)):
    #     left = tree[i].get_left()
    #     right = tree[i].get_right()
    #     if left != None and right != None:
    #         for j in range(left, right + 1):
    #             dot3.edge(str(i), str(j))
    #     elif left != None:
    #         dot3.edge(str(i), str(left))
    #     elif right != None:
    #         dot3.edge(str(i), str(right))
    # dot3.render(directory='doctest-output')
    # print('Время построения дерева решений равно：%f' % (t2 - t1))
    # print('Время классификации тестовой выборки равно：%f' % (t3 - t2))
    # print('Точность классификации：%f' % score)
    # print('Параметр установленный на min_sample_leaf：%d' % min_sample_leaf)
    app = QApplication(sys.argv)
    ex = MainWindow.Example()
    sys.exit(app.exec_())