# Tensors
# Warm-up: numpy
# Before introducing PyTorch, we will first implement the network using numpy.
# Numpy provides an n-dimensional array object, and many functions for manipulating these arrays. Numpy is a generic framework for scientific computing; it does not know anything about computation graphs, or deep learning, or gradients. However we can easily use numpy to fit a two-layer network to random data by manually implementing the forward and backward passes through the network using numpy operations.

# -*- coding: utf-8 -*-
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)    # 64 x 1000
y = np.random.randn(N, D_out)   # 64 x 10

# Randomly initialize weights
w1 = np.random.randn(D_in, H)   # 1000 x 100
w2 = np.random.randn(H, D_out)  # 100 x 10

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)                   # 64 x 1000 . 1000 x 100 = 64 x 100
    h_relu = np.maximum(h, 0)       # 64 x 100
    y_pred = h_relu.dot(w2)         # 64 x 100 . 100 x 10 = 64 x 10

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)        # 64 x 10
    grad_w2 = h_relu.T.dot(grad_y_pred)     # 100 x 64 . 64 x 10 = 100 x 10 (y = h2w2, dw = x'.dy)
    grad_h_relu = grad_y_pred.dot(w2.T)     # 64 x 10 . 10 x 100 = 64 x 100 (y = h2w2, dh2 = dy.w2')
    grad_h = grad_h_relu.copy()             # 64 x 100 (h2 = max(h1, 0), dh1 = dh2 and dh1 = 0 if h1 < 0
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)               # 1000 x 64 . 64 x 100 (h1 = xw1, dw = x'.dh1)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2