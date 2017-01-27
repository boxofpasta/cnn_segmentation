import tensorflow as tf
import numpy as np
import BatchLoader

def init_rand_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev= 0.2))

def init_rand_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

class Model:

    def __init__(self):

        loader = BatchLoader.BatchLoader()

        # define graph
        x, y = loader.get_batch(20)

        # first conv layer
        W1 = init_rand_weight([10, 10, 3, 64])
        b1 = init_rand_bias([64])
        conv1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME') + b1, name='features_1')
        out1 = tf.nn.max_pool(conv1, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_1')

        # second conv layer
        W2 = init_rand_weight([5, 5, 64, 128])
        b2 = init_rand_bias([128])
        conv2 = tf.nn.relu(tf.nn.conv2d(out1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2, name='features_2')
        out2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')

        # third conv layer
        W3 = init_rand_weight([5, 5, 128, 256])
        b3 = init_rand_bias([256])
        conv3 = tf.nn.relu(tf.nn.conv2d(out2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3, name='features_3')
        out3 = tf.nn.max_pool(conv3, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_3')

        # fourth conv layer
        W4 = init_rand_weight([5, 5, 256, 22])
        conv4 = tf.nn.relu(tf.nn.conv2d(out3, W4, strides=[1, 1, 1, 1], padding='SAME'), name='features_4')
        out4 = tf.nn.max_pool(conv4, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_4')

        # define loss
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out4, y))
        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    def get_train_step(self):
        return self.train_step

    def get_loss(self):
        return self.loss