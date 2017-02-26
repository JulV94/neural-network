#!/usr/bin/python3

import os
import pickle
from math import floor
import tools.neural3layers as n3
from tools.load_img import load_img


def train(iterat, layer_size_factor):
    in_arr = []
    out_arr = []
    train_true_dir = "train/true/"
    train_false_dir = "train/false/"
    for img in os.listdir(train_true_dir):
        in_arr.append(load_img(train_true_dir + img)[1])
        out_arr.append([1])
    for img in os.listdir(train_false_dir):
        in_arr.append(load_img(train_false_dir + img)[1])
        out_arr.append([0])

    return n3.train(in_arr, out_arr, iterat, floor(len(in_arr[0])/layer_size_factor))


if __name__ == '__main__':
    print("training...")
    net = train(10000, 200)
    print("Saving network...")
    with open("network.pkl", "wb") as f:
        pickle.dump(net, f)
    print("Network :")
    print(net)
