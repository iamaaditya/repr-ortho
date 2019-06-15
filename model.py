from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange
import tensorflow as tf
from functools import reduce
slim = tf.contrib.slim

def convnet(image, nb_classes):
    ''' vanilla convnet '''
    output = image 
    ini = tf.truncated_normal_initializer(stddev=0.04)
    conv_activations = []
    for layer_num, filter_size in enumerate([32]*3):
        scope = 'conv'+str(layer_num)
        output = slim.conv2d(output, filter_size, [5, 5], scope=scope, weights_initializer=ini)
        # add filter level mask for the network
        betas = tf.placeholder(dtype=tf.float32, shape=(filter_size), name=scope+'/yeta')
        tf.add_to_collection('yeta', betas)
        print(output.shape, betas.shape)
        output = output * betas
        conv_activations.append(output)
        output = slim.max_pool2d(output, [3, 3], stride=2, padding='SAME', data_format='NHWC', scope='pool'+str(layer_num))
	if layer_num >= 7:
	    # ideally BN should not be added to see RePr effects but it is harder to train deeper network without BN
	    output = slim.batch_norm(output, scope='batchnorm'+str(layer_num))
    final_shape = output.get_shape().as_list()[1:]
    number_of_dense = reduce(lambda a, b: a * b, final_shape)
    output_conv = tf.reshape(output, [-1, number_of_dense])
    output_dense = tf.layers.dense( output_conv, nb_classes, activation=None, name='dense1', kernel_initializer=ini)
    return output_dense, conv_activations
