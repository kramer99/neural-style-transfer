# -*- coding: utf-8 -*-

import imageio
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from keras import backend as K
from PIL import Image
from vgg19 import build_model
import image_utils

def get_activations(input_image):
    f = K.function([model.input], [model.get_layer('conv2_1').output,
                                   model.get_layer('conv3_1').output,
                                   model.get_layer('conv4_1').output,
                                   model.get_layer('conv5_1').output])
    return f([input_image])

def content_cost(a_C, a_G):
    _, n_H, n_W, n_C = a_C.shape
    return K.sum(K.pow(a_C - a_G, 2)) / (4 * n_H * n_W * n_C)

def style_cost_layer(a_S, a_G):
    _, n_H, n_W, n_C = a_S.shape
    a_S_reshaped = K.reshape(a_S, (n_H * n_W, n_C))
    a_G_reshaped = K.reshape(a_G, (n_H * n_W, n_C))
    G_S = K.dot(K.transpose(a_S_reshaped), a_S_reshaped)      # gram matrix (style)
    G_G = K.dot(K.transpose(a_G_reshaped), a_G_reshaped)      # gram matrix (generated)
    return K.sum(K.pow(G_S - G_G, 2)) / (2 * n_H * n_W * n_C)**2

def style_cost(activations_S, activations_G):
    cost = 0
    layer_coefficients = [.25, .25, .25, .25]
    for i in range(len(activations_S)):
        a_S = activations_S[i]
        a_G = activations_G[i]
        cost = cost + (layer_coefficients[i] * style_cost_layer(a_S, a_G))
    return cost

def total_cost(J_content, J_style, alpha = 1, beta = 4):
    return alpha * J_content + beta * J_style

def compute_cost_and_gradients(activations_C, activations_S, activations_G, input_tensor):
    J_content = content_cost(activations_C, activations_G[1])     # conv3_1
    J_style = style_cost(activations_S, activations_G)
    J_total = total_cost(J_content, J_style)
    grads_out = K.gradients(J_total, model.input)
    f = K.function([model.input], [J_total, J_content, J_style, grads_out[0]])
    J, J_C, J_S, grads = f([input_tensor])
    print('J: %12d  content: %5d  style: %10d' %(J, J_C, J_S))
    J_history.append(J)
    return J, grads

content_image = imageio.imread('elephant.jpg')
content_image = image_utils.reshape_and_normalize_image(content_image)
input_shape = content_image.shape[-3:]
#style_image = Image.open('styles/hr_giger_biomechanicallandscape_II.jpg')
style_image = Image.open('starry_night-resized.jpg')
style_image = style_image.convert('RGB').resize(input_shape[:-1])
style_image = image_utils.reshape_and_normalize_image(np.asarray(style_image))
generated_image = image_utils.generate_noise_image(content_image)

model = build_model(input_shape)
J_history = []

# these activations can be evaluated once now, as they won't change
activations_C = get_activations(content_image)[1]   # just layer conv3_1 
activations_S = get_activations(style_image)

# these activations will be evaluated on each iteration within fmin_l_bfgs_b
activations_G = [model.get_layer('conv2_1').output,
                 model.get_layer('conv3_1').output,
                 model.get_layer('conv4_1').output,
                 model.get_layer('conv5_1').output]

# to avoid computing cost and gradients twice, we cache the gradients in this 
# structure when f(x) is called, and reference it within df(x)
class cache:    
    grads = None
    
def f(x):
    x = x.reshape((content_image.shape))
    J, grads = compute_cost_and_gradients(activations_C, activations_S,
                                          activations_G, x)
    cache.grads = grads
    return J

def df(x):
    assert cache.grads is not None
    grads = cache.grads
    cache.grads = None
    # fmin_l_bfgs_b only handles 1D arrays
    grads = grads.flatten()
    # fmin_l_bfgs_b needs this to be float64 for some undocumented weird reason
    grads = grads.astype(np.float64)
    return grads
    
# It only occurred to me mid way through this project that I would not be able to use
# Keras's built in optimizers, as the loss function needs to be supplied explicitly,
# hence why using scipy's L-BFGS optimizer here.
# The optimizer will find pixel values that lower the scalar output of the cost function 'f'
generated_image, min_val, info = fmin_l_bfgs_b(f, generated_image.flatten(), fprime=df, maxfun=40)
    
generated_image = generated_image.reshape(content_image.shape)
    
image_utils.save_image('output/output_%d.png' %(time.time()), generated_image)    
#plt.imshow(image_utils.convert(generated_image))
plt.plot(np.arange(0,len(J_history)), J_history, label='J')
