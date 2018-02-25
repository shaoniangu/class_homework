#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import neighbors


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print tip + '正确率：', np.mean(acc)
# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
# iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = 'iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    model = Pipeline([
        ('ss', StandardScaler()),
        ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=3))])
    model = model.fit(x_train, y_train)
    y_hat = model.predict(x_test)      # 测试数据
    show_accuracy(y_hat, y_test, '测试集')