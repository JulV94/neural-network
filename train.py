#!/usr/bin/python3

import neural3layers as n3
import os


def load_img(img_path):
    with open(img_path) as f:
        data = [x for x in f if not x.startswith('#')]  # remove comments
    p = data.pop(0)  # P thing
    dim = tuple(map(int, data.pop(0).split()))
    arr = []
    for line in data:
        for c in line.strip():
            arr.append(int(c))
    return dim, arr


def train(iterat):
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

    return n3.train(in_arr, out_arr, iterat)


def test(network):
    in_arr = []
    test_dir = "test/"
    for img in os.listdir(test_dir):
        in_arr.append(load_img(test_dir + img)[1])
    results = dict(zip(os.listdir(test_dir), n3.use(in_arr, network)))
    return results


if __name__ == '__main__':
    print("training...")
    net = train(10000)
    print("testing...")
    res = test(net)
    print("Network :")
    print(net)
    print("Test results :")
    print(res)
