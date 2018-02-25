# -*-coding:utf-8-*-
# __Author__=Youzhi Gu
# Learn Python at Zhejiang University


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import matplotlib.pyplot as plt


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = 'path/to/model/'
MODEL_NAME = 'model.ckpt'
train_accuracy_history_index = []
train_accuracy_history = []

def train(mnist):
    x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer([REGULARAZTION_RATE])
    y = mnist_inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples/BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x : xs, y_ : ys})

            if i % 100 == 0:
                print('After %d training step(s),loss on training batch is %g.'%(step, loss_value))
                train_accuracy_history.append(1-loss_value)
                train_accuracy_history_index.append(step)
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
train(mnist)
fig = plt.figure()
fig = fig.add_subplot(1,1,1)
fig.plot(train_accuracy_history_index,train_accuracy_history)
fig.set_xlabel('Steps(×100)')
fig.set_ylabel('Accuracy(1)')
fig.set_title('Rate of convergence of Mnist in two-layer CNN')
plt.show()