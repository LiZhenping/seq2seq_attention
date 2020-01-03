import numpy as np
from numpy.random import rand


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return x * (1.0 - x)


def tanh_grad(x):
    return 1 - x ** 2


def initalize(dim, init_range):
    return rand(*dim) * init_range


def zeros(*dim):
    return np.zeros(dim)


def ones(*dim):
    return np.ones(dim)
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    return sm
def softmax_grad(s): 
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x. 
    # s.shape = (1, n) 
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    num = s-s*s
    return num