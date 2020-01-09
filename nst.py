# -*- coding: utf-8 -*-

import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from keras import backend as K
from vgg19 import build_model
import image_utils

def content_cost(a_C, a_G):
    n_H, n_W, n_C = K.int_shape(a_C)
    return K.sum(K.pow(a_C - a_G, 2)) / (4 * n_H * n_W * n_C)

def style_cost(a_S, a_G):
    n_H, n_W, n_C = K.int_shape(a_S)
    a_S_reshaped = K.reshape(a_S, (n_H * n_W, n_C))
    a_G_reshaped = K.reshape(a_G, (n_H * n_W, n_C))
    G_S = K.dot(K.transpose(a_S_reshaped), a_S_reshaped)      # gram matrix (style)
    G_G = K.dot(K.transpose(a_G_reshaped), a_G_reshaped)      # gram matrix (generated)
    return K.sum(K.pow(G_S - G_G, 2)) / K.cast(K.pow(2 * n_H * n_W * n_C, 2), dtype='float32')

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    return alpha * J_content + beta * J_style

def compute_cost_and_gradients(a_C, a_S, a_G, input_tensor):
    J_content = content_cost(a_C, a_G)
    J_style = style_cost(a_S, a_G)
    J_total = total_cost(J_content, J_style)
    grads_out = K.gradients(J_total, model.input)
    f = K.function([model.input], [J_total, J_content, J_style, grads_out[0]])
    J, J_C, J_S, grads = f([input_tensor])
    print('J content: ', J_C, ' J style: ', J_S, ' J total: ', J)
    return J, grads

model = build_model()

style_image = imageio.imread('starry_night-resized.jpg')
style_image = image_utils.reshape_and_normalize_image(style_image)
content_image = imageio.imread('elephant.jpg')
content_image = image_utils.reshape_and_normalize_image(content_image)
generated_image = image_utils.generate_noise_image(content_image)
plt.imshow(generated_image[0])

# make this a unit test
# a_S = K.random_normal_variable(shape=(4, 4, 3), mean=1, scale=4, seed=1)
# a_G = K.random_normal_variable(shape=(4, 4, 3), mean=1, scale=4, seed=1)
# J_style = style_cost(a_S, a_G)
# print(K.eval(J_style))

# we represent the three images as three input samples so that we can calculate costs and gradients via a single forward/back pass
input_tensor = K.concatenate([content_image, style_image, generated_image], axis=0)
layer = model.get_layer('conv3_1').output   # TODO: more layers than just conv3_1
a_C = layer[0, :, :, :]
a_S = layer[1, :, :, :]
a_G = layer[2, :, :, :]

# to avoid computing cost and gradients twice, we cache the gradients in this 
# structure when f(x) is called, and reference it within df(x)
class cache:    
    grads = None

num_iterations = 1
for i in range(num_iterations):
    print('iteration: ', i)
    
    def f(x):
        J, grads = compute_cost_and_gradients(a_C, a_S, a_G, input_tensor)
        cache.grads = grads
        return J
    
    def df(x):
        assert cache.grads is not None
        grads = cache.grads
        cache.grads = None
        # on first iteration, generated image will be the last input sample of three.
        # on subsequent iterations, it will be the last of one
        grads = grads[-1]
        # fmin_l_bfgs_b only handles 1D arrays
        grads = grads.flatten()
        # fmin_l_bfgs_b needs this to be float64 for some undocumented weird reason
        grads = grads.astype(np.float64)
        return grads
        
    # this should update the generated image
    generated_image, min_val, info = fmin_l_bfgs_b(f, generated_image.flatten(), fprime=df, maxfun=10)
    print('loss:', min_val)
    
    # the activations for style and content will not change, so from now on we 
    # reduce the input to just the image being generated
    input_tensor = generated_image
    
    fname = 'at_iteration_%d.png' % i
    image_utils.save_image(fname, generated_image)
