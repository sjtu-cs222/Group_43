import tensorflow as tf

class Discriminator(object):
    
    def __init__(self):
        self.dis_inter_layer_dim = config.dis_inter_layer_dim
        self.init_kernel = config.init_kernel
    
    def discriminator_network(self, x_inp, is_training=False, getter=None, reuse=False):
        """ Discriminator architecture in tensorflow

        Discriminates between real data and generated data

        Args:
            x_inp (tensor): input data for the encoder. [batch_size x sequnce_length x hidden_size]
            reuse (bool): sharing variables or not

        Returns:
            logits (tensor): last activation layer of the discriminator (shape 1)
            intermediate_layer (tensor): intermediate layer for feature matching

        """
        batch_size = x_inp.shape[0]
        x_inp = tf.reshape(x_inp,[batch_size, -1])
        with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):

            name_net = 'layer_1'
            with tf.variable_scope(name_net):
                net = tf.layers.dense(x_inp,
                                      units=256,
                                      kernel_initializer=init_kernel,
                                      name='fc')
                net = leakyReLu(net)
                net = tf.layers.dropout(net, rate=0.2, name='dropout',
                                      training=is_training)

            name_net = 'layer_2'
            with tf.variable_scope(name_net):
                net = tf.layers.dense(net,
                                      units=128,
                                      kernel_initializer=init_kernel,
                                      name='fc')
                net = leakyReLu(net)
                net = tf.layers.dropout(net, rate=0.2, name='dropout',
                                      training=is_training)

            name_net = 'layer_3'
            with tf.variable_scope(name_net):
                net = tf.layers.dense(net,
                                      units=dis_inter_layer_dim,
                                      kernel_initializer=init_kernel,
                                      name='fc')
                net = leakyReLu(net)
                net = tf.layers.dropout(net,
                                        rate=0.2,
                                        name='dropout',
                                        training=is_training)

            intermediate_layer = net

            name_net = 'layer_4'
            with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=1,
                                  kernel_initializer=init_kernel,
                                  name='fc')

        net = tf.squeeze(net)

        return net, intermediate_layer

def leakyReLu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
