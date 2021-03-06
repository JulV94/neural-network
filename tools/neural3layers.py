#!/usr/bin/python3

import numpy as np


# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))


def train(in_array, out_array, iteration, layer_size):
    # input dataset
    x = np.array(in_array)

    # output dataset
    y = np.array(out_array)

    # seed random numbers to make calculation
    # deterministic (just a good practice)
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    syn0 = 2*np.random.random((len(in_array[0]), layer_size)) - 1
    syn1 = 2*np.random.random((layer_size, 1)) - 1

    for iterat in range(iteration):

        # Feed forward through layers 0, 1, and 2
        l0 = x
        l1 = nonlin(np.dot(l0, syn0))
        l2 = nonlin(np.dot(l1, syn1))

        # how much did we miss the target value?
        l2_error = y - l2

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2, deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1, deriv=True)

        # update weights
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)
        if 100 * iterat % iteration == 0:
            print(iterat)
    return [syn0, syn1]


def use(in_array, network, rounded=False):

    l1 = nonlin(np.dot(np.array(in_array), network[0]))
    l2 = nonlin(np.dot(l1, network[1]))
    res = l2.tolist()

    if rounded:
        for i in range(len(res)):
            res[i] = round(res[i][0])
    else:
        for i in range(len(res)):
            res[i] = res[i][0]

    return res
