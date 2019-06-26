import torch
import torch.nn as nn
import subprocess
import pandas as pd
import pickle
from sklearn.metrics import f1_score
import numpy 
import numpy as np

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    if len(preds.shape) == 1:
        rounded_preds = torch.round(torch.sigmoid(preds))
    else:
        rounded_preds = preds.argmax(1)
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()
    
    if isinstance(criterion, nn.CrossEntropyLoss):
        dtype = torch.LongTensor
    elif isinstance(criterion, nn.BCEWithLogitsLoss):
        dtype = torch.FloatTensor

    for i, batch in enumerate(iterator):

        optimizer.zero_grad()
        device = batch.text.device
        labels = batch.label.type(dtype).to(device)
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if i > len(iterator):
            break


    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# def get_f1(pred_choice,target):
#     TP = TN = FN = FP = 0
#     TP += ((pred_choice == 1) & (target.data == 1)).cpu().sum()
#     TN += ((pred_choice == 0) & (target.data == 0)).cpu().sum()
#     FN += ((pred_choice == 0) & (target.data == 1)).cpu().sum()
#     FP += ((pred_choice == 1) & (target.data == 0)).cpu().sum()
#     p = TP / (TP + FP)
#     r = TP / (TP + FN)
#     F1 = 2 * r * p / (r + p)
#     return F1


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    model.eval()
    
    if isinstance(criterion, nn.CrossEntropyLoss):
        dtype = torch.LongTensor
    elif isinstance(criterion, nn.BCEWithLogitsLoss):
        dtype = torch.FloatTensor

    with torch.no_grad():

        for i, batch in enumerate(iterator):
            
            device = batch.text.device
            labels = batch.label.type(dtype).to(device)
            predictions = model(batch.text).squeeze(1)
            if len(predictions.shape) == 1:
                rounded_preds = torch.round(torch.sigmoid(predictions))
            else:
                rounded_preds = predictions.argmax(1)
            f1_predictions = rounded_preds.to('cpu')
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            f1_labels = batch.label.type(dtype).to('cpu')
            #F1 = get_f1(predictions,labels)
            F1 = f1_score(f1_labels,f1_predictions)
            # F1 = torch.from_numpy(f1)
            
            epoch_f1 += F1.item()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if i > len(iterator):
                break

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)