#!/usr/bin/python3

import os
import pickle
import tools.neural3layers as n3
from tools.load_img import load_img


def use(network, rounded=False):
    in_arr = []
    use_dir = "predict/"
    for img in os.listdir(use_dir):
        in_arr.append(load_img(use_dir + img)[1])
    return dict(zip(os.listdir(use_dir), n3.use(in_arr, network, rounded=rounded)))

print("Loading network...")
net = pickle.load(open("network.pkl", "rb"))
print("Network :")
print(net)

res = use(net, False)
res_rounded = use(net, True)
print("Results :")
print(res)
print(res_rounded)
