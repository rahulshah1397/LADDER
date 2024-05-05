#!/usr/bin/env python3
import os
import sys, getopt
import numpy as np
import torch
import toml
from tqdm import tqdm
from utils.dataset import Dataset, Samples
from utils.train import train
from utils.utils import save_results, path_checker, write_summary
from utils.model import Sequence
from utils.inference import Inference

config = toml.load("config.toml")
models = {"sequence":Sequence}
device = torch.device("cuda")

def inference(pairs, path):
    model = Sequence()
    model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
    model.to(device)
    train_dataset = Dataset(mode = "train", pairs = pairs)
    linear_span_results = Inference(model, train_dataset).get(np.linspace(0.1, 10, 1000))
    save_results(linear_span_results, os.path.join(path, f"linear_span.csv"))

def main():
    assert torch.cuda.is_available()
    output_directory, pairs = parse_arguments()

    #load datasets.
    train_dataset = Dataset(mode = "train", pairs = pairs)
    val_dataset = Dataset(mode = "val", pairs = pairs)

    #Train and save trained model and statistics.
    model = Sequence()
    model.to(device)
    model, loss_track = train(model, train_dataset, val_dataset)
    np.save(os.path.join(output_directory, "losses.npy"), loss_track)
    torch.save(model.state_dict(), os.path.join(output_directory, "model.pth"))

    #Run inference on pantheon+ and linear span.
    test_dataset = Dataset(mode = "test", pairs = pairs)
    pantheon_plus_results = Inference(model, test_dataset).test_set_prediction()
    save_results(pantheon_plus_results, os.path.join(output_directory, "pantheon_plus.csv"))
    inference(pairs, output_directory)

def parse_arguments():
    pairs = config['dataset']['PAIRS']
    output_directory = None
    argumentList = sys.argv[1:]
    long_options, options = ["help", "pairs=", "output="], "hp:o:"
    arguments, values = getopt.getopt(argumentList, options, long_options)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--Help"):
            print (config['help'])
            exit(0)
        elif currentArgument in ("-p", "--pairs"):
            pairs = int(currentValue)
            assert 1 <= pairs <= 1000
        elif currentArgument in ("-o", "--output"):
            output_directory = path_checker(currentValue)
        else:
            raise ValueError("Invalid input, see python main.py --help")
    if output_directory is None:
        output_directory = path_checker()
    write_summary(output_directory, pairs)
    return output_directory, pairs

if __name__ == "__main__":
    main()
