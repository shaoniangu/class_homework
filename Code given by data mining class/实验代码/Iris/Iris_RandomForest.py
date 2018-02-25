#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    path = 'iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x_prime, y = np.split(data, (4,), axis=1)
    train_x, test_x, train_y, test_y = train_test_split(x_prime,y,test_size=0.2,random_state=0)
    clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=4)
    rf_clf = clf.fit(train_x,train_y)
    y_hat = rf_clf.predict(test_x)
    acc = u'准确率：%.2f%%' % (100 * np.mean(y_hat == test_y.ravel()))
    print acc
