#!/usr/bin/python3

import os
import pickle
import tools.neural3layers as n3
from tools.load_img import load_img


def test(network, rounded=False):
    in_arr = []
    test_dir = "test/"
    for img in os.listdir(test_dir):
        in_arr.append(load_img(test_dir + img)[1])
    results = dict(zip(os.listdir(test_dir), n3.use(in_arr, network, rounded=rounded)))
    return results


if __name__ == '__main__':
    print("Loading network...")
    net = pickle.load(open("network.pkl", "rb"))
    print("Network :")
    print(net)
    print("testing...")
    res = test(net, False)
    res_rounded = test(net, True)
    print("Test results :")
    print(res)
    print(res_rounded)
