# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print tip + '正确率：', np.mean(acc)

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
if __name__ == "__main__":
    path = 'iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    clf = KMeans(n_clusters=3, init='k-means++')
    clf.fit(x_train, y_train.ravel())
    y_hat = clf.predict(x_test)
    y_test=y_test.ravel()
    m_hat = np.array([np.mean(x_test[y_hat == i], axis=0) for i in range(3)])
    m = np.array([np.mean(x_test[y_test == i], axis=0) for i in range(3)])
    order = pairwise_distances_argmin(m, m_hat, axis=1, metric='euclidean')
    print order
    n_sample = y_test.size
    n_types = 3
    change = np.empty((n_types, n_sample), dtype=np.bool)
    for i in range(n_types):
        change[i] = y_hat == order[i]
    for i in range(n_types):
        y_hat[change[i]] = i
    acc = u'准确率：%.2f%%' % (100 * np.mean(y_hat == y_test))
    print acc


