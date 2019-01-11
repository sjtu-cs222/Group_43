import tensorflow as tf
from configuration import generator_config

class Generator(object):
    
    def __init__(config):
        self.data_window = config.data_window
        self.feature_size = config.feature_size
        self.num_classes = config.num_classes
        self.filter_size = config.filter_size
        self.num_filters = config.num_filters
        self.hidden_size = self.num_filters
        self.dense_size = config.dense_size
        self.num_layers = config.num_layers
        self.l2_reg_lambda = config.l2_reg_lambda
        self.init_kernel = config.init_kernel
        
    def generate_fake_sample(self, z_inp, is_training=False, getter=None, reuse=False, batch_size):
        """ Generator architecture in tensorflow
        Generates data from the latent space
        Args:
            z_inp (tensor): variable in the latent space
            reuse (bool): sharing variables or not
        Returns:
            (tensor): last activation layer of the generator

        """
        with tf.variable_scope('generate_fake_sample', reuse=reuse, custom_getter=getter):

            name_net = 'layer_1'
            with tf.variable_scope(name_net):
                net = tf.layers.dense(z_inp,
                                      units=64,
                                      kernel_initializer=init_kernel,
                                      name='fc')
                net = tf.nn.relu(net, name='relu')

            name_net = 'layer_2'
            with tf.variable_scope(name_net):
                net = tf.layers.dense(net,
                                      units=128,
                                      kernel_initializer=init_kernel,
                                      name='fc')
                net = tf.nn.relu(net, name='relu')

            name_net = 'layer_4'
            with tf.variable_scope(name_net):
                net = tf.layers.dense(net,
                                      units=self.feature_size,
                                      kernel_initializer=init_kernel,
                                      name='fc')
            tf.reshape(net, [batch_size, self.data_window, self.feature_size]
            return net
            
    def build(self):
        """Create all network for pretraining, adversarial training and sampling"""
        self.build_pretrain_network()
        #self.build_adversarial_network()
        
    def build_input(self, name):
        assert name in ['pretrain', 'adversarial']
        if name == 'pretrain':
            self.pretrain_batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
            self.pretrain_input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.data_window, self.feature_size], name='pretrain_input_x')
            self.pretrain_input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='pretrain_input_y')
            self.pretrain_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='pretrain_keep_prob')
        elif name == 'adversarial':
            self.adversarial_input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.data_window, self.feature_size], name='adversarial_input_x')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.adversarial_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='adversarial_prob')
    
    def build_pretrain_network(self):
        """ Buid pretrained network
        Output:
            self.pretrain_loss
        """
        self.build_input(name="pretrain")
        self.pretrain_l2_loss = tf.constant(0.0)
        inputs = tf.expand_dims(self.pretrain_input_x, -1)
        inputs = tf.nn.dropout(inputs, keep_prob=self.pretrain_keep_prob)
        self.pretrain_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        
        with tf.variable_scope("teller"):
            # First convolutional layer
            with tf.variable_scope('conv1-%s' % self.filter_size):
                filter_shape = [self.filter_size, self.feature_size, 1, self.num_filters]
                W1 = tf.get_variable('weights1', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b1 = tf.get_variable('biases1', [self.num_filters], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.conv2d(inputs, W1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
                h1 = tf.nn.tanh(tf.nn.bias_add(conv1, b1), name='tanh1')
            #First max pooling layer
            with tf.variable_scope('max-pooling1'):
                pooled1 = tf.nn.max_pool(h1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool1")
            # Second convolutional layer
            with tf.variable_scope('conv2-%s' % self.filter_size):
                filter_shape = [self.filter_size, self.feature_size, self.num_filters, self.num_filters]
                W2 = tf.get_variable('weights2', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b2 = tf.get_variable('biases2', [self.num_filters], initializer=tf.constant_initializer(0.0))
                conv2 = tf.nn.conv2d(pooled1, W2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
                h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2), name='tanh2')
            # Second max pooling layer
            with tf.variable_scope('max-pooling2'):
                pooled2 = tf.nn.max_pool(h2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool2")
            
            rnn_inputs = tf.squeeze(pooled2, [2])
            with tf.variable_scope('LSTM'):
                cell = tf.contrib.rnn.LSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.pretrain_keep_prob)
                cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)
                self._initial_state = cell.zero_state(self.pretrain_batch_size, dtype=tf.float32)
                outputs, state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=self._initial_state)
                self.pretrain_final_state = state
            # Softmax output layer
            with tf.name_scope('softmax'):
                softmax_w1 = tf.get_variable('softmax_w1', shape=[self.hidden_size, self.dense_size], dtype=tf.float32)
                softmax_b1 = tf.get_variable('softmax_b1', shape=[self.dense_size], dtype=tf.float32)
                softmax_w2 = tf.get_variable('softmax_w2', shape=[self.dense_size, self.num_classes], dtype=tf.float32)
                softmax_b2 = tf.get_variable('softmax_b2', shape=[self.num_classes], dtype=tf.float32)
                # L2 regularization for output layer
                self.pretrain_l2_loss += tf.nn.l2_loss(softmax_w1)
                self.pretrain_l2_loss += tf.nn.l2_loss(softmax_b1)
                self.pretrain_l2_loss += tf.nn.l2_loss(softmax_w2)
                self.pretrain_l2_loss += tf.nn.l2_loss(softmax_b2)
                # logits
                dense1 = tf.nn.tanh(tf.matmul(self.pretrain_final_state[self.num_layers - 1].h, softmax_w1) + softmax_b1, name='dense1')
                self.pretrain_logits = tf.matmul(dense1, softmax_w2) + softmax_b2
                predictions = tf.nn.softmax(self.pretrain_logits)
                self.pretrain_predictions = tf.argmax(predictions, 1, name='predictions')
            # Loss
            with tf.name_scope('loss'):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.pretrain_input_y, logits=self.pretrain_logits)
                self.pretrain_loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.pretrain_l2_loss
            # Accuracy
            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.pretrain_predictions, self.pretrain_input_y)
                self.pretrain_correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
                self.pretrain_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    def build_adversarial_network(self):
        """ Buid adversarial network
        Output:
            self.gen_loss_adv
        """
        self.build_input(name="adversarial")
        #self.pretrain_l2_loss = tf.constant(0.0)
        inputs = tf.expand_dims(self.adversarial_input_x, -1)
        #inputs = tf.nn.dropout(inputs, keep_prob=self.adversarial_keep_prob)
        
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            # First convolutional layer
            with tf.variable_scope('conv1-%s' % self.filter_size):
                filter_shape = [self.filter_size, self.feature_size, 1, self.num_filters]
                W1 = tf.get_variable('weights1', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b1 = tf.get_variable('biases1', [self.num_filters], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.conv2d(inputs, W1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
                h1 = tf.nn.tanh(tf.nn.bias_add(conv1, b1), name='tanh1')
            #First max pooling layer
            with tf.variable_scope('max-pooling1'):
                pooled1 = tf.nn.max_pool(h1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool1")
            # Second convolutional layer
            with tf.variable_scope('conv2-%s' % self.filter_size):
                filter_shape = [self.filter_size, self.feature_size, self.num_filters, self.num_filters]
                W2 = tf.get_variable('weights2', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b2 = tf.get_variable('biases2', [self.num_filters], initializer=tf.constant_initializer(0.0))
                conv2 = tf.nn.conv2d(pooled1, W2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
                h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2), name='tanh2')
            # Second max pooling layer
            with tf.variable_scope('max-pooling2'):
                pooled2 = tf.nn.max_pool(h2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool2")
            
            rnn_inputs = tf.squeeze(pooled2, [2])
            with tf.variable_scope('LSTM'):
                cell = tf.contrib.rnn.LSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.adversarial_keep_prob)
                cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)
                #self._initial_state = cell.zero_state(self.pretrain_batch_size, dtype=tf.float32)
                #outputs, state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=self._initial_state)
                outputs, state = tf.nn.dynamic_rnn(cell, rnn_inputs)
                self.adversarial_final_state = state
            # Softmax output layer
            with tf.name_scope('softmax'):
                softmax_w1 = tf.get_variable('softmax_w1', shape=[self.hidden_size, self.dense_size], dtype=tf.float32)
                softmax_b1 = tf.get_variable('softmax_b1', shape=[self.dense_size], dtype=tf.float32)
                softmax_w2 = tf.get_variable('softmax_w2', shape=[self.dense_size, self.num_classes], dtype=tf.float32)
                softmax_b2 = tf.get_variable('softmax_b2', shape=[self.num_classes], dtype=tf.float32)
                '''
                # L2 regularization for output layer
                self.pretrain_l2_loss += tf.nn.l2_loss(softmax_w1)
                self.pretrain_l2_loss += tf.nn.l2_loss(softmax_b1)
                self.pretrain_l2_loss += tf.nn.l2_loss(softmax_w2)
                self.pretrain_l2_loss += tf.nn.l2_loss(softmax_b2)
                '''
                # logits
                dense1 = tf.nn.tanh(tf.matmul(self.adversarial_final_state[self.num_layers - 1].h, softmax_w1) + softmax_b1, name='dense1')
                self.adversarial_logits = tf.matmul(dense1, softmax_w2) + softmax_b2
                predictions = tf.nn.softmax(self.logits)
                self.adversarial_predictions = tf.argmax(predictions, 1, name='predictions')
            # Loss
            with tf.name_scope('loss'):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.pretrain_input_y, logits=self.logits)
                self.adversarial_loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.pretrain_l2_loss
            # Accuracy
            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.pretrain_predictions, self.pretrain_input_y)
                self.pretrain_correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
                self.pretrain_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
                
    def classifier_network(self, data):
        '''
        inputs: real or fake data, [batch_size, data_window, feature_size]
        outputs: lstm cells [batch_size, lstm length, hidden_size]
        '''
        inputs = tf.expand_dims(data, -1)
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            # First convolutional layer
            with tf.variable_scope('conv1-%s' % self.filter_size):
                filter_shape = [self.filter_size, self.feature_size, 1, self.num_filters]
                W1 = tf.get_variable('weights1', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b1 = tf.get_variable('biases1', [self.num_filters], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.conv2d(inputs, W1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
                h1 = tf.nn.tanh(tf.nn.bias_add(conv1, b1), name='tanh1')
            #First max pooling layer
            with tf.variable_scope('max-pooling1'):
                pooled1 = tf.nn.max_pool(h1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool1")
            # Second convolutional layer
            with tf.variable_scope('conv2-%s' % self.filter_size):
                filter_shape = [self.filter_size, self.feature_size, self.num_filters, self.num_filters]
                W2 = tf.get_variable('weights2', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b2 = tf.get_variable('biases2', [self.num_filters], initializer=tf.constant_initializer(0.0))
                conv2 = tf.nn.conv2d(pooled1, W2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
                h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2), name='tanh2')
            # Second max pooling layer
            with tf.variable_scope('max-pooling2'):
                pooled2 = tf.nn.max_pool(h2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool2")
            
            rnn_inputs = tf.squeeze(pooled2, [2])
            with tf.variable_scope('LSTM'):
                cell = tf.contrib.rnn.LSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.adversarial_keep_prob)
                cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)
                #self._initial_state = cell.zero_state(self.pretrain_batch_size, dtype=tf.float32)
                #outputs, state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=self._initial_state)
                outputs, state = tf.nn.dynamic_rnn(cell, rnn_inputs)
            
            # Softmax output layer
            with tf.name_scope('softmax'):
                softmax_w1 = tf.get_variable('softmax_w1', shape=[self.hidden_size, self.dense_size], dtype=tf.float32)
                softmax_b1 = tf.get_variable('softmax_b1', shape=[self.dense_size], dtype=tf.float32)
                softmax_w2 = tf.get_variable('softmax_w2', shape=[self.dense_size, self.num_classes], dtype=tf.float32)
                softmax_b2 = tf.get_variable('softmax_b2', shape=[self.num_classes], dtype=tf.float32)
                # logits
                dense1 = tf.nn.tanh(tf.matmul(self.adversarial_final_state[self.num_layers - 1].h, softmax_w1) + softmax_b1, name='dense1')
                self.adversarial_logits = tf.matmul(dense1, softmax_w2) + softmax_b2
            # Loss
            with tf.name_scope('loss'):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.constant(0, shape=[batch_size]), logits=self.logits)
                loss = tf.reduce_mean(losses)
        return outputs, loss


















