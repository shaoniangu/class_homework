# -*-coding:utf-8-*-
# __Author__=Youzhi Gu
# Learn Python at Zhejiang University

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x : mnist.validatoin.images,y_ : mnist.validation.labels}
        y = mnist_inference.inference(x,None)

        correct_predicion = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predicion,tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DACAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                print('After %s traing step(s),validation accuracy = %g'%(global_step,accuracy_score))

            else:
                print('No checkpoint file found')
                return
            time.sleep(EVAL_INTERVAL_SECS)

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
evaluate(mnist)