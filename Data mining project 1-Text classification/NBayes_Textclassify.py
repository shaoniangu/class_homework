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
# This code realize step 3 to step 5 using NBayes
# 输入
# 1. 训练文件的分词持久化后的.dat文件
# 2. 测试文件的分词持久化后的.dat文件
# 3. 停用词表.txt文件
# 输出
# 1. 训练文件的词空间向量
# 2. 测试文件的词空间向量
# 3. 分类的结果评估指标值


import sys
import os
from sklearn import feature_extraction
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt


# 以rb方式打开文件
def readfile(path):
    fp = open(path, 'rb')
    content = fp.read()
    fp.close()
    return content


# 以wb方式写文件
def savefile(savepath,content):
    fp = open(savepath,'wb')
    fp.write(content)
    fp.close()


# 读持久化bunch文件
def readbunchobj(path):
    file_obj = open(path,'rb')
    bunch = pickle.load(file_obj, encoding='iso-8859-1')
    # iso-8859-1 是一种字符集，这里要用‘iso-8859-1’加载，否则是乱码
    file_obj.close()
    return bunch

# 写持久化bunch文件
def writebunchobj(path,bunchobj):
    file_obj = open(path,'wb')
    pickle.dump(bunchobj, file_obj) # 这里不需要指定字符集，默认为0，以文本形式序列化
    file_obj.close()

# 定义结果评估函数
def metrics_result(actual,predict):
    print("精度：{0:.3f}".format(metrics.precision_score(actual,predict,average="weighted")))
    print("召回：{0:.3f}".format(metrics.recall_score(actual, predict,average="weighted")))
    print("F1-score：{0:.3f}".format(metrics.f1_score(actual, predict,average="weighted")))

# 载入停用词表，规避停用词对分类的影响
stopword_path = "train_wordbag/hlt_stop_words.txt"
stpwrdlst = readfile(stopword_path).splitlines()


#####################训练样本的TF-IDF计算#######################################################

# 载入训练分词后持久化序列
path = 'train_wordbag/train_set.dat'
bunch = readbunchobj(path)

# 生成训练文本的词空间向量，以训练文本bunch的内部参数定义词空间向量的内部参数
Train_tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames,
                   tdm=[], vocabulary={})

# 生成TF-IDF向量空间模型
vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
# 统计每个词语的TF-IDF值
transformer = TfidfTransformer()
# 计算词频矩阵
Train_tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
# 单独保存一下字典
Train_tfidfspace.vocabulary=vectorizer.vocabulary_
# 持久化TF-IDF向量词袋
space_path = "train_wordbag/Train_tfidfspace.dat"
writebunchobj(space_path,Train_tfidfspace)

#####################训练样本的TF-IDF计算#######################################################
# 读分词后测试样本
path = "test_wordbag/test_set.dat"
bunch = readbunchobj(path)
# 建立测试集的向量空间
Test_tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label,filenames = bunch.filenames,
                  tdm=[],vocabulary={})
# 读训练样本词向量
trainbunch = readbunchobj("train_wordbag/Train_tfidfspace.dat")
# 用训练样本的词向量生成向量空间模型
vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                             vocabulary=trainbunch.vocabulary)
# 计算TF-IDF值
transformer = TfidfTransformer()
# 用训练样本的词空间向量模型计算测试样本的词频矩阵
Test_tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
# 字典保存
Test_tfidfspace.vocabulary = trainbunch.vocabulary
# 保存测试TF-IDF向量空间
test_space_path = "test_wordbag/Test_tfidfspace.dat"
writebunchobj(test_space_path,Test_tfidfspace)


# 导入文件
train_path = "train_wordbag/Train_tfidfspace.dat"
train_set = readbunchobj(train_path)

test_path = "test_wordbag/Test_tfidfspace.dat"
test_set = readbunchobj(test_path)

#####################分类器训练和预测########################################################
# NBayes训练器训练
clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)
# NBayes训练器预测
predicted = clf.predict(test_set.tdm)

#######################分类结果评估##########################################################
total = len(predicted)
rate= 0
rate_history=[]
for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
        rate +=1
        print(file_name,"实际类别：", flabel, '\t'+"预测类别：", expct_cate)
print("error_rate:",float(rate)*100/float(total),"%")
metrics_result(test_set.label,predicted)