# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim


def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
    return net








if __name__ == '__main__':
    weights = slim.model_variable('weights',
                                  shape=[10, 10, 3, 3],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  regularizer=slim.l2_regularizer(0.05),
                                  device='/CPU:0')

    # Model Variables
    model_variables = slim.get_model_variables()

    # Regular variables
    my_var = slim.variable('my_var',
                           shape=[20, 1],
                           initializer=tf.zeros_initializer())

    regular_variables_and_model_variables = slim.get_variables()

    my_model_variable = CreateViaCustomCode()

    # Letting TF-Slim know about the additional variable.
    slim.add_model_variable(my_model_variable)

    # plain TensorFlow code
    input = ...
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)

    # corresponding TF-Slim code
    input = ...
    net = slim.conv2d(input, 128, [3, 3], scope='conv1_1')

    with tf.Session() as sess:
        sess.run(weights)
        print(weights)
        print(sess.run(weights))

    pass
