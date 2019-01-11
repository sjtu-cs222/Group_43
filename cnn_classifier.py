# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class cnn_clf(object):
    """
    A C-LSTM classifier
    Reference: A C-LSTM Neural Network for Text Classification
    """
    def __init__(self, config):
        self.data_window = config.data_window
        self.feature_size = config.feature_size
        self.num_classes = config.num_classes
        self.filter_size = config.filter_size
        self.num_filters = config.num_filters
        self.l2_reg_lambda = config.l2_reg_lambda

        # Placeholders
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.data_window, self.feature_size], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        #expand_dims
        inputs = tf.expand_dims(self.input_x, -1)
        #inputs [batch_size x data_window x feature_size x 1]

        # Input dropout
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        # First convolutional layer
        with tf.variable_scope('conv1-%s' % self.filter_size):
            filter_shape = [self.filter_size, self.feature_size, 1, self.num_filters]
            W1 = tf.get_variable('weights1', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b1 = tf.get_variable('biases1', [self.num_filters], initializer=tf.constant_initializer(0.0))
            # Convolution
            conv1 = tf.nn.conv2d(inputs, W1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
            #conv1 [batch_size x 60 x 1 x num_filters]
            # Activation function
            h1 = tf.nn.tanh(tf.nn.bias_add(conv1, b1), name='tanh1')
            
        # Second convolutional layer
        with tf.variable_scope('conv1-%s' % self.filter_size):
            filter_shape = [self.filter_size, self.feature_size, self.num_filters, self.num_filters]
            W2 = tf.get_variable('weights2', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b2 = tf.get_variable('biases2', [self.num_filters], initializer=tf.constant_initializer(0.0))
            # Convolution
            conv2 = tf.nn.conv2d(h1, W2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
            #conv2 [batch_size x 60 x 1 x num_filters]
            # Activation function
            h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2), name='tanh2')
            
        #First max pooling layer
        with tf.variable_scope('max-pooling1'):
            pooled1 = tf.nn.max_pool(h2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool1")
        #pooled1 [batch_size x 30 x 1 x num_filters]

        # Third convolutional layer
        with tf.variable_scope('conv3-%s' % self.filter_size*2):
            filter_shape = [self.filter_size, self.feature_size, self.num_filters, self.num_filters*2]
            W3 = tf.get_variable('weights3', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('biases3', [self.num_filters*2], initializer=tf.constant_initializer(0.0))
            # Convolution
            conv3 = tf.nn.conv2d(pooled1, W3, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
            #conv3 [batch_size x 30 x 1 x 64]
            # Activation function
            h3 = tf.nn.tanh(tf.nn.bias_add(conv3, b3), name='tanh3')

        # Forth convolutional layer
        with tf.variable_scope('conv4-%s' % self.filter_size*2):
            filter_shape = [self.filter_size, self.feature_size, self.num_filters*2, self.num_filters*2]
            W4 = tf.get_variable('weights4', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b4 = tf.get_variable('biases4', [self.num_filters*2], initializer=tf.constant_initializer(0.0))
            # Convolution
            conv4 = tf.nn.conv2d(h3, W4, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
            #conv4 [batch_size x 30 x 1 x 64]
            # Activation function
            h4 = tf.nn.tanh(tf.nn.bias_add(conv4, b4), name='tanh4')

        # Second max pooling layer
        with tf.variable_scope('max-pooling2'):
            pooled2 = tf.nn.max_pool(h4, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool2")
        #pooled2 [batch_size x 15 x 1 x num_filters*2]

        # Fifth convolutional layer
        with tf.variable_scope('conv5-%s' % self.filter_size*4):
            filter_shape = [self.filter_size, self.feature_size, self.num_filters*2, self.num_filters*4]
            W5 = tf.get_variable('weights5', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b5 = tf.get_variable('biases5', [self.num_filters*4], initializer=tf.constant_initializer(0.0))
            # Convolution
            conv5 = tf.nn.conv2d(pooled2, W5, strides=[1, 1, 1, 1], padding='SAME', name='conv5')
            #conv5 [batch_size x 15 x 1 x 128]
            # Activation function
            h5 = tf.nn.tanh(tf.nn.bias_add(conv5, b5), name='tanh5')

        # Sixth convolutional layer
        with tf.variable_scope('conv6-%s' % self.filter_size*4):
            filter_shape = [self.filter_size, self.feature_size, self.num_filters*4, self.num_filters*4]
            W6 = tf.get_variable('weights6', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b6 = tf.get_variable('biases6', [self.num_filters*4], initializer=tf.constant_initializer(0.0))
            # Convolution
            conv6 = tf.nn.conv2d(h5, W6, strides=[1, 1, 1, 1], padding='SAME', name='conv6')
            #conv6 [batch_size x 15 x 1 x 128]
            # Activation function
            h6 = tf.nn.tanh(tf.nn.bias_add(conv6, b6), name='tanh6')

        flat_length = (self.data_window // 4) * self.num_filters*4
        h6 = h6.reshape([-1, flat_length])

        # Softmax output layer
        with tf.name_scope('softmax'):
            softmax_w1 = tf.get_variable('softmax_w1', shape=[flat_length, self.num_filters*16], dtype=tf.float32)
            softmax_b1 = tf.get_variable('softmax_b1', shape=[self.num_filters*16], dtype=tf.float32)
            softmax_w2 = tf.get_variable('softmax_w2', shape=[self.num_filters*16, self.num_filters*8], dtype=tf.float32)
            softmax_b2 = tf.get_variable('softmax_b2', shape=[self.num_filters*8], dtype=tf.float32)
            softmax_w3 = tf.get_variable('softmax_w3', shape=[self.num_filters*8, self.num_classes], dtype=tf.float32)
            softmax_b3 = tf.get_variable('softmax_b3', shape=[self.num_classes], dtype=tf.float32)

            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w1)
            self.l2_loss += tf.nn.l2_loss(softmax_b1)
            self.l2_loss += tf.nn.l2_loss(softmax_w2)
            self.l2_loss += tf.nn.l2_loss(softmax_b2)
            self.l2_loss += tf.nn.l2_loss(softmax_w3)
            self.l2_loss += tf.nn.l2_loss(softmax_b3)

            # logits
            dense1 = tf.nn.tanh(tf.matmul(h6, softmax_w1) + softmax_b1, name='dense1')
            dense2 = tf.nn.tanh(tf.matmul(dense1, softmax_w2) + softmax_b2, name='dense2')
            self.logits = tf.matmul(dense2, softmax_w3) + softmax_b3
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1, name='predictions')

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
