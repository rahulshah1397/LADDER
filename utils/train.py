#!/usr/bin/env python3
import numpy as np
import torch,toml,re
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau as Scheduler
from utils.utils import MSE, DKL

config = toml.load("config.toml")
device = torch.device("cuda")


def train(model, train_dataset, val_dataset):
    """ Function to train and evaluate model."""
    train_dataloader = DataLoader(train_dataset, batch_size = config['params']['BATCH_SIZE'], shuffle = True)
    optimizer = AdamW(model.parameters(), lr = config['params']['LR'], eps = 1e-8, weight_decay = 1e-4)
    epochs = config['params']['EPOCHS']
    scheduler = Scheduler(optimizer, mode = "min", factor = 0.5, patience = 2)
    counter, patience, patience_limit, train_loss, best = 1, 0, 6, 0.0, float('inf')
    loss_track = []
    model.zero_grad()
    for epoch_number in range(int(epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for i,batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            loss, predicted_mean, predicted_var = model(*batch)
            train_loss += loss.item()
            if counter%1 == 0:
                loss_track.append(train_loss/counter)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            counter += 1
            epoch_iterator.set_description(f"Epoch [{epoch_number+1}/{epochs}] Loss : {train_loss/counter:.6f}")
            epoch_iterator.refresh()
        mse, dkl = evaluate(model, device, val_dataset)
        print(f"[Validation Set] | D_kl = {dkl:.6f} | MSE = {mse:.6f} |")
        scheduler.step(dkl)
        if mse < best:
            patience = 0
            best = mse
            best_state = model.state_dict()
        else:
            patience +=1
        if patience >= patience_limit:
            model.load_state_dict(best_state)
            print("Early stopping.")
            break
    return model, np.array(loss_track)


def evaluate(model, device, dataset):
    """ Function to evaluate the model at epoch end"""
    eval_dataloader = DataLoader(dataset, batch_size = config['params']['BATCH_SIZE'], shuffle = False)
    predicted_means, predicted_vars, target_means, target_vars = [], [], [], []
    model.eval()
    for i,batch in enumerate(tqdm(eval_dataloader, desc = "Evaluating")):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            predicted_mean, predicted_var = model(*batch)
        predicted_means.extend(predicted_mean.detach().cpu().tolist())
        predicted_vars.extend(predicted_var.detach().cpu().tolist())
        target_vars.extend(batch[-1].detach().cpu().tolist())
        target_means.extend(batch[-2].detach().cpu().tolist())
    predicted_means, predicted_vars = np.array(predicted_means), np.array(predicted_vars)
    target_means, target_vars = np.array(target_means), np.array(target_vars)
    mse = MSE(target_means, predicted_means)
    dkl = DKL(target_means, target_vars, predicted_means, predicted_vars)
    return mse, dkl 
