# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.models import Sequential

def build_model():

    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        Credit: deeplearning.ai
        """
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b
    
    def _set_weights_on_layer(keras_layer, vgg_layer):
        W, b = _weights(vgg_layer, model.layers[keras_layer].name)
        b = np.reshape(b, (b.size))
        model.layers[keras_layer].set_weights([W, b])

    vgg = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
    vgg_layers = vgg['layers']
    
    model = Sequential([
        #Convolution2D(64, (3, 3), name='conv1_1', padding='same', activation='relu', input_shape=(300,400,3)),
        Convolution2D(64, (3, 3), name='conv1_1', padding='same', activation='relu', input_shape=(224,224,3)),
        Convolution2D(64, (3, 3), name='conv1_2', padding='same', activation='relu'),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='avgpool1', padding='same'),
        Convolution2D(128, (3, 3), name='conv2_1', padding='same', activation='relu'),
        Convolution2D(128, (3, 3), name='conv2_2', padding='same', activation='relu'),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='avgpool2', padding='same'),
        Convolution2D(256, (3, 3), name='conv3_1', padding='same', activation='relu'),
        Convolution2D(256, (3, 3), name='conv3_2', padding='same', activation='relu'),
        Convolution2D(256, (3, 3), name='conv3_3', padding='same', activation='relu'),
        Convolution2D(256, (3, 3), name='conv3_4', padding='same', activation='relu'),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='avgpool3', padding='same'),
        Convolution2D(512, (3, 3), name='conv4_1', padding='same', activation='relu'),
        Convolution2D(512, (3, 3), name='conv4_2', padding='same', activation='relu'),
        Convolution2D(512, (3, 3), name='conv4_3', padding='same', activation='relu'),
        Convolution2D(512, (3, 3), name='conv4_4', padding='same', activation='relu'),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='avgpool4', padding='same'),
        Convolution2D(512, (3, 3), name='conv5_1', padding='same', activation='relu'),
        Convolution2D(512, (3, 3), name='conv5_2', padding='same', activation='relu'),
        Convolution2D(512, (3, 3), name='conv5_3', padding='same', activation='relu'),
        Convolution2D(512, (3, 3), name='conv5_4', padding='same', activation='relu'),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='avgpool5', padding='same'),
    ])
    
    _set_weights_on_layer(0, 0)
    _set_weights_on_layer(1, 2)
    _set_weights_on_layer(3, 5)
    _set_weights_on_layer(4, 7)
    _set_weights_on_layer(6, 10)
    _set_weights_on_layer(7, 12)
    _set_weights_on_layer(8, 14)
    _set_weights_on_layer(9, 16)
    _set_weights_on_layer(11, 19)
    _set_weights_on_layer(12, 21)
    _set_weights_on_layer(13, 23)
    _set_weights_on_layer(14, 25)
    _set_weights_on_layer(16, 28)
    _set_weights_on_layer(17, 30)
    _set_weights_on_layer(18, 32)
    _set_weights_on_layer(19, 34)
    
    #model.summary()
    return model

