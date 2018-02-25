# -*-coding:utf-8-*-
# __Author__ = Lu Lv
# Learn Python at Zhejiang University
# Theme: Classify Chinese Text With Labels
# Step：
# 1. Divide sentences into words using JIEBA packge
# 2. Data persistence using Bunch data structure
# 3. Establish word vector space
# 4. Train classifier
# 5. Test classifier and showing resuls
# This code realize step 3 to step 5 using SVM



import sys
import os
# 引入Bunch类
from sklearn.datasets.base import Bunch
# 引入持久化类
import pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # 导入线性SVM算法
import numpy as np
from sklearn import metrics


# 读取文件
def readfile(path):
    fp = open(path, "rb")
    content = fp.read()
    fp.close()
    return content


# 计算分类精度：
def metrics_result(actual, predict):
    print("精度：{0:.3f}".format(metrics.precision_score(actual, predict, average="weighted")))
    print("召回：{0:.3f}".format(metrics.recall_score(actual, predict, average="weighted")))
    print("F1-score：{0:.3f}".format(metrics.f1_score(actual, predict, average="weighted")))


# 读取bunch对象
def readbunchobj(path):
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj, encoding='iso-8859-1')
    file_obj.close()
    return bunch


# 写入bunch对象
def writebunchobj(path, bunchobj):
    file_obj = open(path, "wb")
    pickle.dump(bunchobj, file_obj)
    file_obj.close()


# 导入训练集
trainpath = "train_wordbag/Train_tfidfspace.dat"
train_set = readbunchobj(trainpath)

# 导入测试集
testpath = "test_wordbag/Test_tfidfspace.dat"
test_set = readbunchobj(testpath)


# 应用线性SVM算法
# 1. 输入词袋向量和分类标签
clf = LinearSVC(penalty="l2", dual=False, tol=1e-4).fit(train_set.tdm, train_set.label)

# 预测分类结果
predicted = clf.predict(test_set.tdm)
total = len(predicted)
rate = 0
for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        rate += 1
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)

# 精度
print("error rate:", float(rate) * 100 / float(total), "%")
print("预测完毕!!!")
metrics_result(test_set.label, predicted)

