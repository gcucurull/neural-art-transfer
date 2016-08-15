# Adapted from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ by Michael Guerzhoy and Davi Frossard, 2016
import numpy as np
import scipy.misc
import tensorflow as tf
import os
import math

def get_model(input_image):
    # define 2dconv
    def conv2d(x, W, stride, padding="SAME", group=1):
        if group == 1:
            return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        else:
            # Make sure that the number of input and output feature maps
            # be divided in N groups
            assert x.get_shape()[-1]%group==0
            assert W.get_shape()[-1]%group==0
            input_groups = tf.split(3, group, x)
            kernel_groups = tf.split(3, group, W)
            output_groups = [tf.nn.conv2d(i, k, strides=[1, stride, stride, 1], padding=padding) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
            return conv

    # define max pooling
    def max_pool(x, k_size, stride, padding="VALID"):
        return tf.nn.max_pool(x, ksize=[1, k_size, k_size, 1], 
                strides=[1, stride, stride, 1], padding=padding)

    def get_type(layer):
        if 'conv' in layer:
            return 'conv'
        if 'lrn' in layer:
            return 'lrn'
        if 'pool' in layer:
            return 'pool'
        if 'fc' in layer:
            return 'fc'
        if 'relu' in layer:
            return 'relu'

    model = {}

    # load weights
    net_data = np.load(os.path.dirname(__file__)+"/bvlc_alexnet.npy").item()

    layers = []
    layers.extend(['conv1', 'relu1', 'lrn1', 'maxpool1'])
    layers.extend(['conv2', 'relu2', 'lrn2', 'maxpool2'])
    layers.extend(['conv3', 'relu3', 'conv4', 'relu4', 'conv5', 'relu5', 'maxpool5'])

    # first input
    current = input_image
    
    for layer in layers:
        layer_type = get_type(layer)

        if layer_type == 'conv':
            W_conv = tf.constant(net_data[layer][0])
            b_conv = tf.constant(net_data[layer][1])
            if layer == 'conv1':
                conv_out = conv2d(current, W_conv, stride=4, padding='SAME', group=1)
            elif layer == 'conv3':
                conv_out = conv2d(current, W_conv, stride=1, padding='SAME', group=1)
            else:
                conv_out = conv2d(current, W_conv, stride=1, padding='SAME', group=2)
            
            current = tf.nn.bias_add(conv_out, b_conv)

        elif layer_type == 'lrn':
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            current = tf.nn.local_response_normalization(current,
                          depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

        elif layer_type == 'pool':
            current = max_pool(current, k_size=3, stride=2, padding="VALID")

        elif layer_type == 'relu':
            current = tf.nn.relu(current)

        model[layer] = current

    return model

    # conv1
    # conv(11,11,3,96) with stride=4, group=1, padding="SAME"

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')

    # conv2
    # conv(5, 5, 96, 256) with stride=1, group=2, padding="SAME"

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  

    # conv3
    # conv(3, 3, 256, 384) with stride=1, group=1 padding="SAME"

    # conv4
    # conv(3, 3, 384, 384) with stride=1, group=2, padding="SAME"

    # conv5
    # conv(3, 3, 384, 256) with stride=1, group=2, padding="SAME"

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')

def preprocess(image):
    mean = np.array([104, 117, 123]) # bgr
    image = image[...,::-1] # rgb to bgr
    image = image - mean
    return image

def postprocess(image):
    mean = np.array([104, 117, 123]) # bgr
    image = image[...,::-1] # bgr to rgb
    image = image + mean
    return image

def content_layers():
    return 'conv4'

def style_layers():
    return ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']