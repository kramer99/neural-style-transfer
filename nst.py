# -*- coding: utf-8 -*-

import scipy.io
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from keras import backend as K
from vgg19 import build_model
import image_utils

def get_layer_activations(input_image, layer_name):
    f = K.function([model.layers[0].input], [model.get_layer(layer_name).output])
    return f([input_image])[0]

def content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.shape
    J_content = np.sum(np.power(a_C - a_G, 2)) / (4 * n_H * n_W * n_C)
    return J_content

def style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.shape
    a_S_reshaped = np.reshape(a_S, (n_H * n_W, n_C))
    a_G_reshaped = np.reshape(a_G, (n_H * n_W, n_C))
    G_S = np.dot(a_S_reshaped.T, a_S_reshaped)      # gram matrix (style)
    G_G = np.dot(a_G_reshaped.T, a_G_reshaped)      # gram matrix (generated)
    J_style = np.sum(np.power(G_S - G_G, 2)) / (np.power(2 * n_H * n_W * n_C, 2))
    return J_style

def total_cost(J_content, J_style, alpha = 1, beta = 10000):
    return alpha * J_content + beta * J_style

model = build_model()

style_image = imageio.imread('starry_night-resized.jpg')
style_image = image_utils.reshape_and_normalize_image(style_image)
content_image = imageio.imread('elephant.jpg')
content_image = image_utils.reshape_and_normalize_image(content_image)
generated_image = image_utils.generate_noise_image(content_image)
plt.imshow(generated_image[0])

# TODO: more layers than just conv3_1
a_C = get_layer_activations(content_image, 'conv3_1')
a_S = get_layer_activations(style_image, 'conv3_1')
#a_G = get_layer_activations(generated_image, 'conv3_1')
#J_content = content_cost(a_C, a_G)
#J_style = style_cost(a_S, a_G)
#J_total = total_cost(J_content, J_style)

num_iterations = 5
for i in range(num_iterations):
    
    print('begin: ', generated_image.shape)
    
    # total cost for this iteration of generated image
    a_G = get_layer_activations(generated_image, 'conv3_1')
    J_content = content_cost(a_C, a_G)
    J_style = style_cost(a_S, a_G)
    J_total = total_cost(J_content, J_style)
    
    # get the gradients of the generated image wrt the total cost
    generated_image_placeholder = K.placeholder(generated_image.shape)
    loss_var = K.variable(0.0)
    J_total_var = K.variable(J_total)   # TODO: J_total is not connected to the graph, hence undifferentiable
    grads_f = K.gradients(J_total_var, generated_image_placeholder)
    
    outputs = [loss_var]
    if isinstance(grads_f, (list, tuple)):
        outputs += grads_f
    else:
        outputs.append(grads_f)    
    f = K.function([generated_image_placeholder], outputs)
    #f = K.function([generated_image_placeholder], [loss_var, grads_f])
    loss, grads = f([generated_image])

    # this should update the generated image 
    generated_image, min_val, info = fmin_l_bfgs_b(loss, generated_image.flatten(), fprime = grads, maxfun=20)
    print('after optimize shape: ', generated_image.shape)    
    print('loss:', min_val)
    
    fname = 'at_iteration_%d.png' % i
    image_utils.save_img(fname, generated_image)
