# -*-coding:utf-8-*-
# __Author__=Youzhi Gu
# Learn Python at Zhejiang University
# 代码任务：
# 去除每一行的分号，否则每行的最后一个数会被当做字符串读入


import numpy as np


file = open('WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt','r')
contents = file.read().replace(';','')
file.close()

file = open('WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt','w')
file.write(contents)
file.close()

a = np.array([1,2,3,4,5,6,7,8,9,10])
train_test_split = np.random.rand(10) < 0.70
print(train_test_split)
print(a[train_test_split],'真的可以用布尔值啊！！！')