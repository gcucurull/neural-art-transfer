import numpy as np
import os
import tensorflow as tf

def get_model(input_image):
    # define 2dconv
    def conv2d(x, W, stride, padding="SAME"):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    # define max pooling
    def max_pool(x, k_size, stride, padding="VALID"):
        # use avg pooling instead, as described in the paper
        return tf.nn.avg_pool(x, ksize=[1, k_size, k_size, 1], 
                strides=[1, stride, stride, 1], padding=padding)

    def get_type(layer):
        if 'conv' in layer:
            return 'conv'
        if 'pool' in layer:
            return 'pool'
        if 'relu' in layer:
            return 'relu'

    model = {}

    # load weights
    net_data = np.load(os.path.dirname(__file__)+"/vgg16_weights.npz")

    layers = []
    layers.extend(['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1'])
    layers.extend(['conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2'])
    layers.extend(['conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'maxpool3'])
    layers.extend(['conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'maxpool4'])
    layers.extend(['conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'maxpool5'])

    # first input
    current = input_image

    for layer in layers:
        layer_type = get_type(layer)

        if layer_type == 'conv':
            W_conv = tf.constant(net_data[layer+'_W'])
            b_conv = tf.constant(net_data[layer+'_b'])
            conv_out = conv2d(current, W_conv, stride=1, padding='SAME')
            
            current = tf.nn.bias_add(conv_out, b_conv)

        elif layer_type == 'pool':
            current = max_pool(current, k_size=2, stride=2, padding="SAME")

        elif layer_type == 'relu':
            current = tf.nn.relu(current)

        model[layer] = current

    return model

def preprocess(image):
    mean = np.array([104, 117, 123]) # bgr
    image = image[...,::-1] # rgb to bgr
    image = image - mean
    return image

def postprocess(image):
    mean = np.array([104, 117, 123]) # bgr
    image = image + mean
    image = image[...,::-1] # bgr to rgb
    return image

def content_layers():
    return 'conv4_2'

def style_layers():
    return ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
