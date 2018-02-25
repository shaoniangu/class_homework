# __Author__ = Lu Lv
# Learn Python at Zhejiang University
# Theme: Classify Chinese Text With Labels
# Step：
# 1. Divide sentences into words using JIEBA packge
# 2. Data persistence using Bunch data structure
# 3. Establish word vector space
# 4. Train classifier
# 5. Test classifier and showing resuls
# This code realize step 2
# 输入
# 1. 训练文件分词后文件
# 2. 测试文件分词后文件（从总分词后文件中随机筛选一些作为分词后文件）
# 输出
# 1. 训练文件分词后持久化文件
# 2. 测试文件分词后持久化文件

from sklearn.datasets.base import Bunch
import pickle
import os
import sys


# 保存至文件
def savefile(savepath, content):
    fp = open(savepath, "w", encoding='utf-8')
    fp.write(content)
    fp.close()


# 读取文件
def readfile(path):
    fp = open(path, "r", encoding='utf-8')
    content = fp.read()
    fp.close()
    return content

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


# 定义训练集的bunch
train_bunch = Bunch(target_name=[],label=[],filenames=[],contents=[])
# 训练集持久化文件保存路径
train_wordbag_path = 'train_wordbag/train_set.dat'
# 训练集分词后文件路径
train_seg_path = 'train_txt_divided/'

#当前文件夹所有文件夹名的列表
catelist = os.listdir(train_seg_path)
#文件夹名就是类别，保存至bunch中
train_bunch.target_name.extend(catelist)
#遍历每个文件夹
for mydir in catelist:
    #拼接文件路径
    class_path = train_seg_path+mydir+'/'
    #得到文件夹中所有文件的文件名
    file_list = os.listdir(class_path)
    # 遍历每个文件夹中的文件
    for file_path in file_list:
        #拼接全路径
        fullname = class_path + file_path
        #将文件名作为标签传个bunch中的标签
        train_bunch.label.append(mydir)
        #当前的文件全路径传给bunch中的路径
        train_bunch.filenames.append(fullname)
        #保存文件词向量
        train_bunch.contents.append(readfile(fullname).strip())

# bunch对象持久化
file_obj = open(train_wordbag_path,'wb')
pickle.dump(train_bunch,file_obj)
file_obj.close()

#定义测试集bunch
test_bunch = Bunch(target_name=[],label=[],filenames=[],contents=[])
# 测试集持久化文件保存路径
test_wordbag_path = 'test_wordbag/test_set.dat'
# 测试集分词后文件路径
test_seg_path = 'test_txt_divided/'

#当前文件夹所有文件夹名的列表
catelist = os.listdir(test_seg_path)
#文件夹名就是类别，保存至bunch中
test_bunch.target_name.extend(catelist)
#遍历每个文件夹
for mydir in catelist:
    #拼接文件路径
    class_path = train_seg_path+mydir+'/'
    #得到文件夹中所有文件的文件名
    file_list = os.listdir(class_path)
    # 遍历每个文件夹中的文件
    for file_path in file_list:
        #拼接全路径
        fullname = class_path + file_path
        #将文件名作为标签传个bunch中的标签
        test_bunch.label.append(mydir)
        #当前的文件全路径传给bunch中的路径
        test_bunch.filenames.append(fullname)
        #保存文件词向量
        test_bunch.contents.append(readfile(fullname).strip())

# bunch对象持久化
file_obj = open(test_wordbag_path, 'wb')
pickle.dump(test_bunch, file_obj)
file_obj.close()

print('模块持久化结束！')