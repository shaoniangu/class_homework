#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import neighbors
from sklearn.lda import LDA
from sklearn.cross_validation import train_test_split

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print tip + '正确率：', np.mean(acc)

if __name__ == "__main__":
    path = 'iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    clf =neighbors.KNeighborsClassifier(algorithm='kd_tree')
    clf.fit(x_train, y_train.ravel())
    y_hat = clf.predict(x_train)
    show_accuracy(y_hat, y_train, '训练集')
    y_hat = clf.predict(x_test)
    show_accuracy(y_hat, y_test, '测试集')

