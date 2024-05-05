#!/usr/bin/env python3
import numpy as np
import torch,toml,re
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import Dataset, Samples

config = toml.load("config.toml")

class Inference:
    """ Class implementing predictions from the trained LSTM model."""

    def __init__(self, model, dataset):
        self.model = model
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.dataset = dataset

    def get(self, X):
        """ Computes predictions for all values of z in an iterable X. 
        Args:
            x : iterable

        Returns:
            res (list[dict]) : predicted mean and variance for every z in X.
        """
        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size = config['params']['SUPPORT'], shuffle = True)
        res = []
        for x in X:
            feature, _, _ = next(iter(dataloader))
            if len(feature.shape) == 3:
                feature[:, -1, 0] = x
                feature[:, -1, 1] = torch.mean(feature[:, :-1,1])
            else:
                feature[:,-1] = x
            feature = feature.to(self.device)
            with torch.no_grad():
                predicted_mean, predicted_var = self.model(feature)
            pred_mean = torch.mean(predicted_mean).detach().cpu().item()+Samples.cdm(x)
            pred_std = torch.sqrt(torch.mean(predicted_var)).detach().cpu().item()
            res.append({"x":x,"predicted_mean":pred_mean,"predicted_std":pred_std})
        return res

    def test_set_prediction(self):
        """ Computes predictions for all values of z in an iterable X. 
        Args:
            x : iterable

        Returns:
            res (list[dict]) : predicted mean and variance for every z in X.
        """
        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size = config['params']['BATCH_SIZE'], shuffle = False)
        target_means, target_vars = {},{}
        predicted_means, predicted_vars = {},{}
        K = config['params']['SUPPORT']
        for ep in tqdm(range(K), desc = "Evaluating pantheon plus"):
            for feature, target_mean, target_var in dataloader:
                if len(feature.shape) == 3:
                    X = feature[:, -1, 0]
                else:
                    X = feature[:,-1]
                batch = feature.to(self.device)
                with torch.no_grad():
                    predicted_mean, predicted_var = self.model(batch)
                predicted_mean = predicted_mean+Samples.cdm(X).to(self.device)
                target_mean = target_mean+Samples.cdm(X)
                predicted_mean, predicted_var = predicted_mean.detach().cpu().tolist(), predicted_var.detach().cpu().tolist()
                target_mean, target_var = target_mean.detach().cpu().tolist(), target_var.detach().cpu().tolist()
                for i,x in enumerate(X.detach().cpu().tolist()):
                    x = f"{x:.6f}"
                    if x in predicted_means:
                        predicted_means[x].append(predicted_mean[i])
                        predicted_vars[x].append(predicted_var[i])
                    else:
                        target_means[x] = target_mean[i]
                        target_vars[x] = target_var[i]
                        predicted_means[x] = [predicted_mean[i]]
                        predicted_vars[x] = [predicted_var[i]]
        res = []
        for x in target_means.keys():
            prediction = {"x":x, "target_mean":target_means[x], "target_std":np.sqrt(target_vars[x])}
            prediction["predicted_mean"] = np.mean(predicted_means[x])
            prediction["predicted_std"] = np.sqrt(np.mean(predicted_vars[x]))
            res.append(prediction)
        return res
