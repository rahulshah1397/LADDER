#!/usr/bin/env python3
import numpy as np
import toml
import os
import datetime

config = toml.load("config.toml")


def save_results(results, filename):
    """ Helper function to write output in csv format """
    fin = []
    keys = list(results[0].keys())
    fin.append("\t".join(keys))
    for res in results:
        fin.append("\t".join([str(res[key]) for key in keys]))
    with open(filename, "w") as f:
        f.writelines("\n".join(fin))


def path_checker(path = None):
    """ Checks if the given output path already exists or creates it. If no
    path name is provided a path is created with time-stamps.
    """
    if path is None:
        path = f"output_{datetime.datetime.now().strftime('%d-%m-%H:%M:%S')}"
    base = config['files']['SAVE_DIR']
    if not os.path.exists(base):
        os.makedirs(base)
    base = os.path.join(base, path)
    if not os.path.exists(base):
        os.makedirs(base)
        return base
    cnt = 1
    while os.path.exists(f"{base}_{cnt}"):
        cnt +=1
    path = f"{base}_{cnt}"
    os.makedirs(path)
    return path


def write_summary(path, pairs):
    """ Helper function to write a summary file in the save directory. """
    doc = f"PAIRS = {pairs}\n"\
          "---------------------------------------------------------\n"\
          "This folder contains model weights, inference results, etc\n"\
          f"for LSTM trained with {pairs} samples as support.\n"\
          "Please check README for more info."
    with open(os.path.join(path, "DETAILS.txt"), "w") as f:
        f.writelines(doc)


def MSE(labels, preds):
    return np.mean(np.square(labels - preds))

def DKL(target_means, target_vars, predicted_means, predicted_vars):
    ls = np.log2(target_vars) - np.log2(predicted_vars) + (predicted_vars + np.square(target_means - predicted_means))/target_vars
    return 0.5*np.mean(ls) - 0.5
