#!/usr/bin/env python
import toml
import torch
import torch.nn as nn

config = toml.load("config.toml")


class Sequence(nn.Module):
    """ LSTM model to predict mean and variance of m(z) """

    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.hidden_dim = config['params']['HIDDEN_DIM']//2
        self.num_layers = config['params']['NUM_LAYERS']
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True, dropout = 0.5)
        self.m = nn.Linear(self.hidden_dim, 2)
        self.act = nn.GELU()

    def forward(self, observations, target_mean = None, target_var = None):
        """ Forward method to calculate loss, mean, variance. Performs forward
        pass through LSTM followed by a MLP on the last hidden state.

        Args:
            observations : torch.tensor
            target_mean : Optional[torch.tensor]
            target_var : Optional[torch.tensor]

        Returns:
            loss (Optional[float]) : The training loss.
            predicted_mean (float) : Predicted mean of m(z).
            predicted_var (float) : Predicted variance of m(z).
        
        """
        x, hidden = self.lstm(observations)
        x = x[:, -1]
        predict = self.m(self.act(x))
        predicted_mean, predicted_var = torch.split(predict, [1,1], dim = -1)
        predicted_var = torch.clamp(torch.abs(predicted_var), 1e-3, 5)
        if self.training:
            return self.Loss(target_mean, target_var, predicted_mean, predicted_var), predicted_mean, predicted_var
        return predicted_mean.squeeze(-1), predicted_var.squeeze(-1)

    def Loss(self, target_mean, target_var, predicted_mean, predicted_var):
        """ Computes KL divergence between the target and predicted gaussian """
        ls = torch.log2(target_var) - torch.log2(predicted_var) + (predicted_var + torch.square(target_mean - predicted_mean))/target_var
        return 0.5*torch.mean(ls) - 0.5
