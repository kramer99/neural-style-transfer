# -*- coding: utf-8 -*-

import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from keras import backend as K
from PIL import Image
from vgg19 import build_model
import image_utils

def get_activations(input_image):
    f = K.function([model.input], [model.get_layer('conv3_1').output,
                                   model.get_layer('conv4_1').output,
                                   model.get_layer('conv5_1').output])
    return f([input_image])

def content_cost(a_C, a_G):
    n_H, n_W, n_C = a_C.shape
    return K.sum(K.pow(a_C - a_G, 2)) / (4 * n_H * n_W * n_C)

def style_cost_layer(a_S, a_G):
    n_H, n_W, n_C = a_S.shape
    a_S_reshaped = K.reshape(a_S, (n_H * n_W, n_C))
    a_G_reshaped = K.reshape(a_G, (n_H * n_W, n_C))
    G_S = K.dot(K.transpose(a_S_reshaped), a_S_reshaped)      # gram matrix (style)
    G_G = K.dot(K.transpose(a_G_reshaped), a_G_reshaped)      # gram matrix (generated)
    return K.sum(K.pow(G_S - G_G, 2)) / K.cast(K.pow(2 * n_H * n_W * n_C, 2), dtype='float32')

def style_cost(activations_S, activations_G):
    cost = 0
    layer_coefficients = [.2, .4, .6]
    for i in range(len(activations_S)):
        a_S = activations_S[i]
        a_G = activations_G[i]
        cost = cost + (layer_coefficients[i] * style_cost_layer(a_S, a_G))
    return cost

def total_cost(J_content, J_style, alpha = 1, beta = 4):
    return alpha * J_content + beta * J_style

def compute_cost_and_gradients(activations_C, activations_S, activations_G, input_tensor):
    J_content = content_cost(activations_C, activations_G[0])     # conv3_1
    J_style = style_cost(activations_S, activations_G)
    J_total = total_cost(J_content, J_style)
    grads_out = K.gradients(J_total, model.input)
    f = K.function([model.input], [J_total, J_content, J_style, grads_out[0]])
    J, J_C, J_S, grads = f([input_tensor])
    print('J: %16d  content: %5d  style: %14d' %(J, J_C, J_S))
    J_history.append(J)
    return J, grads

model = build_model()

J_history = []

#style_image = Image.open('styles/hr_giger_biomechanicallandscape_II.jpg')
style_image = Image.open('starry_night-resized.jpg')
style_image = style_image.resize((224,224))
style_image = image_utils.reshape_and_normalize_image(np.asarray(style_image))
content_image = imageio.imread('elephant.jpg')
content_image = image_utils.reshape_and_normalize_image(content_image)
generated_image = image_utils.generate_noise_image(content_image)

# these activations can be evaluated once now, as they won't change
activations_C = get_activations(content_image)[0]   # just layer conv3_1 
activations_C = activations_C.reshape((56,56,256))

activations_S = get_activations(style_image)
activations_S[0] = activations_S[0].reshape((56,56,256))
activations_S[1] = activations_S[1].reshape((28,28,512))
activations_S[2] = activations_S[2].reshape((14,14,512))

# these activations will be evaluated on each iteration within fmin_l_bfgs_b
activations_G = [model.get_layer('conv3_1').output,
                 model.get_layer('conv4_1').output,
                 model.get_layer('conv5_1').output]

# to avoid computing cost and gradients twice, we cache the gradients in this 
# structure when f(x) is called, and reference it within df(x)
class cache:    
    grads = None
    
def f(x):
    x = x.reshape((1, image_utils.CONFIG.IMAGE_HEIGHT,
                   image_utils.CONFIG.IMAGE_WIDTH, 
                   image_utils.CONFIG.COLOR_CHANNELS))
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
# hence why using scikit-learn's L-BFGS optimizer here.
# The optimizer will find pixel values that lower the scalar output of the cost function 'f'
generated_image, min_val, info = fmin_l_bfgs_b(f, generated_image.flatten(), fprime=df, maxfun=30)
    
generated_image = generated_image.reshape((1, image_utils.CONFIG.IMAGE_HEIGHT,
                                           image_utils.CONFIG.IMAGE_WIDTH, 
                                           image_utils.CONFIG.COLOR_CHANNELS))
    
image_utils.save_image('output.png', generated_image)    
#plt.imshow(image_utils.convert(generated_image))
plt.plot(np.arange(0,len(J_history)), J_history, label='J')
