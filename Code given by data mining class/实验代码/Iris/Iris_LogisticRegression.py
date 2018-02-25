#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    path = u'iris.data'  # 数据文件路径

    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    print data
    # 将数据的0到3列组成x，第4列得到y
    x, y = np.split(data, (4,), axis=1)
    # train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=0)

    # 为了可视化，仅使用前两列特征
    # x = x[:, :2]
    # #
    # print x
    # print y
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
    # 等价形式
    lr = Pipeline([('sc', StandardScaler()),
                        ('clf', LogisticRegression()) ])
    lr.fit(train_x, train_y.ravel())
    #
    # # 画图
    # N, M = 500, 500     # 横纵各采样多少个值
    # x1_min, x1_max = test_x[:, 0].min(), test_x[:, 0].max()   # 第0列的范围
    # x2_min, x2_max = test_x[:, 1].min(), test_x[:, 1].max()   # 第1列的范围
    # t1 = np.linspace(x1_min, x1_max, N)
    # t2 = np.linspace(x2_min, x2_max, M)
    # x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
    # x_test = np.stack((x1.flat, x2.flat), axis=1)   # 测试点
    #
    # cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    # cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    # y_hat = lr.predict(x_test)                  # 预测值
    # y_hat = y_hat.reshape(x1.shape)                 # 使之与输入的形状相同
    # plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)     # 预测值的显示
    # plt.scatter(test_x[:, 0], test_x[:, 1], c=test_y, edgecolors='k', s=50, cmap=cm_dark)    # 样本的显示
    # plt.xlabel('petal length')
    # plt.ylabel('petal width')
    # plt.xlim(x1_min, x1_max)
    # plt.ylim(x2_min, x2_max)
    # plt.grid()
    # plt.savefig('2.png')
    # plt.show()

    # 训练集上的预测结果
    y_hat = lr.predict(test_x)
    test_y = test_y.reshape(-1)
    result = y_hat == test_y
    print y_hat
    print result
    acc = np.mean(result)
    print '准确度: %.2f%%' % (100 * acc)
