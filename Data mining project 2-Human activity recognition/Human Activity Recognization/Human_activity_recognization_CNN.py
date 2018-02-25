# -*-coding:utf-8-*-
# __Author__=Youzhi Gu
# Learn Python at Zhejiang University
# Data Mining Project 2
# Applying CNN to solve the human activity prediction using Dataset Actitracker
# Step:
# 1.处理数据文件以符合CNN的输入要求
# 2.划分数据为测试集和训练集两部分，用训练集训练CNN网络
# 3.给定测试集一个时间序列的数据，预测人类在这个时间序列中的活动类别


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf


'''
文件加载与预处理环节
'''
# pd.read_csv 读取以逗号分隔的文件，header=None表示没有表头，
# 否则默认以文件的第一行作为表头
def read_data(file_path):
    # 每一列的标签
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data


# 标准化数据
def feature_normalize(dataset):
    # 求均值
    mu = np.mean(dataset, axis=0)
    # 求方差
    sigma = np.std(dataset, axis=0)
    # 标准化
    return (dataset - mu) / sigma


# 画坐标轴
def plot_axis(ax, x, y, title):
    # 描点
    ax.plot(x, y)
    # 设置标题
    ax.set_title(title)
    # 看不见框架
    ax.xaxis.set_visible(False)
    # 设置y坐标轴范围
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    # 设置x坐标轴范围
    ax.set_xlim([min(x), max(x)])
    # 网格可见
    ax.grid(True)


# 根据数据data作图
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


# 加载文件并标准化
dataset = read_data('WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
dataset['z-axis'] = feature_normalize(dataset['z-axis'])

'''
# 按人类活动类别分别显示数据
for activity in np.unique(dataset["activity"]):
    subset = dataset[dataset["activity"] == activity][:180]
    plot_activity(activity, subset)
'''

'''
文件加载与预处理环节结束
'''


'''
CNN数据准备环节
'''
def windows(data, size):
    start = 0
    while start < data.count():
        # 返回的是一个生成器，当有大量的只要读一次数据的时候，就用yield
        yield start, start + size
        start += (size // 2)    # 说实话，没懂


# 分段信号
def segment_signal(data, window_size = 90):
    # 空数组，每个数组的大小为window_size * 3
    segments = np.empty((0, window_size,3))
    # 空数组
    labels = np.empty((0))

    for (start, end) in windows(data["timestamp"], window_size):
        x = data["x-axis"][start:end]
        # print(shape(x))
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if(len(dataset["timestamp"][start:end]) == window_size):
            # dstack 横向拼接，vstack纵向拼接
            segments = np.vstack([segments, np.dstack([x,y,z])])
            # scipy.stats.mode()计算众数
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
    return segments, labels


# 将数据划分为训练集和测试集
segments, labels = segment_signal(dataset)
# get_dummies 生成一个特征map，详看收藏的网页，不用activity对应的字符串作为标签，将标签数字化更好
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
# 按维度重塑矩阵
reshaped_segments = segments.reshape(len(segments), 1,90, 3)
# 生成随机数然后和0.7比较得到布尔值，其实就是70%的True，%30的False
train_test_split = np.random.rand(len(reshaped_segments)) < 0.70

train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

'''
CNN数据准备环节结束
'''

'''
CNN环节
'''
# 定义输入数据维度为1*90*3
input_height = 1
input_width = 90
num_labels = 6      # 分类标签个数
num_channels = 3    # 通道个数，x,y,z

# 定义batch大小
batch_size = 10
# 定义核的大小
kernel_size = 60
# 定义深度
depth = 60
# 定义隐层个数
num_hidden = 1000

# 定义学习率
learning_rate = 0.0001
# 训练轮数
training_epochs = 5

# //是整除的意思
total_batchs = reshaped_segments.shape[0] // batch_size


'''
生成计算图
'''

# 初始化权重变量函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 初始化偏置变量函数
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

# 卷积网络框架
def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')

# 卷积过程
def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))

# 最大池化过程
def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')


# 定义placeholder变量缓存区
X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])

# 第一层卷积层、池化层、第二层卷积层
c = apply_depthwise_conv(X, kernel_size, num_channels, depth)   #kernel_size = 60，depth = 60，num_channels = 3
p = apply_max_pool(c, 20, 2)
c = apply_depthwise_conv(p, 6, depth * num_channels, depth // 10)

# 重塑矩阵为1维，为全连层做准备
shape = c.get_shape().as_list()
c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

# 全连输入层变量设置
f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth // 10), num_hidden])
f_biases_l1 = bias_variable([num_hidden])
# 全连层激活函数
f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1), f_biases_l1))

# 全连输出层变量设置
out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
# softmax层
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

# 定义损失函数
loss = -tf.reduce_sum(Y * tf.log(y_))
# 定义优化决策
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# 预测计算图定义
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
# 精度计算图定义
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 历史代价值保存
cost_history = np.empty(shape=[1], dtype=float)


'''
开始会话，开始训练和预测
'''

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(training_epochs):
        for b in range(total_batchs):
            # 计算batch的下标起始点
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            # 训练batch切片
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            # 训练、损失值计算
            _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, c)
            # 输出损失、精度
        print("Epoch: ", epoch, " Training Loss: ", c, " Training Accuracy: ",
              session.run(accuracy, feed_dict={X: train_x, Y: train_y}))
    # 测试集精度
    print("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))

'''
CNN环节结束
'''