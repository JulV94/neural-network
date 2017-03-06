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


def test(network, rounded=False):
    in_arr = []
    test_dir_true = "test/true/"
    test_dir_false = "test/false/"
    for img in os.listdir(test_dir_true):
        in_arr.append(load_img(test_dir_true + img)[1])
    true_results = dict(zip(os.listdir(test_dir_true), n3.use(in_arr, network, rounded=rounded)))
    in_arr = []
    for img in os.listdir(test_dir_false):
        in_arr.append(load_img(test_dir_false + img)[1])
    false_results = dict(zip(os.listdir(test_dir_false), n3.use(in_arr, network, rounded=rounded)))
    correct = 0
    incorrect = 0
    for val in true_results.values():
        if val < 0.5:
            incorrect += 1
        else:
            correct += 1
    for val in false_results.values():
        if val < 0.5:
            correct += 1
        else:
            incorrect += 1
    accuracy = correct/(correct+incorrect)
    return accuracy, true_results, false_results


def autotune(iterat_step, layer_size_factor_step):
    best = [None, 0, None, None, 0, 0]  # [net, accuracy, true_dict, false_dict, it, lsf]
    it = iterat_step
    while True:
        good = [None, 0, None, None, it, 0]  # [net, accuracy, true_dict, false_dict, it, lsf]
        lsf = layer_size_factor_step
        while True:
            net = train(it, lsf)
            result = test(net, False)
            if result[0] > good[1]:
                good[0] = net
                good[1] = result[0]
                good[2] = result[1]
                good[3] = result[2]
                good[5] = lsf
                lsf += layer_size_factor_step
            else:
                break
        if good[1] > best[1]:
            best = good
            it += iterat_step
        else:
            break
    return best

if __name__ == '__main__':
    auto = False
    if auto:
        print("Autotune activated")
        net, acc, true_res, false_res, it, lsf = autotune(1000, 50)
        print("Saving network...")
        with open("network.pkl", "wb") as f:
            pickle.dump(net, f)
        print("Network :")
        print(net)
        print("Result of true :")
        print(true_res)
        print("Result of false :")
        print(false_res)
        print("Accuracy : ", acc)
        print("Optimal parameters : ")
        print("Iterations : ", it)
        print("Layer size factor : ", lsf)
    else:
        print("Autotune deactivated")
        print("training...")
        net = train(10000, 200)
        print("Testing...")
        acc, true_res, false_res = test(net, False)
        print("Saving network...")
        with open("network.pkl", "wb") as f:
            pickle.dump(net, f)
        print("Network :")
        print(net)
        print("Result of true :")
        print(true_res)
        print("Result of false :")
        print(false_res)
        print("Accuracy : ", acc)
