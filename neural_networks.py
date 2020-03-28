#!/usr/bin/python
# -*- coding: utf-8 -*-
import utils
import torch
import pandas as pd
import numpy as np

# parameters
# nombre d'echantillons
size = (5000, 1002)

# charger les données nécessaires
X_train, y_train, X_test, y_test = utils.load_data(size)
X_final= utils.load_test_data()

d = X_train.shape[1]

# convert into tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test  = torch.tensor(X_test).float()
y_test  = torch.tensor(y_test).float()

X_final = torch.tensor(X_final).float()

# move to cuda if available
if torch.cuda.is_available():
    X_train = X_train.to('cuda')
    y_train = y_train.to('cuda')
    X_test  = X_test.to('cuda')
    y_test  = y_test.to('cuda')

# définition du modèle
class Modele(torch.nn.Module):
    def __init__(self, d):
        super(Modele, self).__init__()

        n1 = d
        n2 = int(d/2)
        n3 = int(d/3)


        self.l1 = torch.nn.Linear(n1, n2)
        self.l2 = torch.nn.Linear(n2, n3)
        self.l3 = torch.nn.Linear(n3, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        out3 = self.sigmoid(self.l3(out2))
        return out3

# prediction
def prediction(f):
    return f.round()

# error rate
def error_rate(labels, preds):
    len_all = len(preds)
    faux    = preds != labels
    return torch.sum(faux) / float(len_all)

# création du modele
modele = Modele(d)

# move to cuda if available
if torch.cuda.is_available():
    modele.to('cuda')

# critère de Loss
criterion = torch.nn.BCELoss()

# optimizer
optimizer = torch.optim.SGD(modele.parameters(), lr=0.01)


for epoch in range(8000):
    f_train = modele(X_train)
    loss = criterion(f_train, y_train)

    if epoch % 100 == 0:
        pred_train = prediction(f_train)
        error_train = error_rate(pred_train, y_train)

        f_test      = modele(X_test)
        pred_test   = prediction(f_test)
        error_test  = error_rate(pred_test, y_test)

        print('epoch: ', epoch, 'loss: ', loss.item(), 'error_train', error_train.item(), 'error_test', error_test.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

f      = modele(X_final)
pred   = prediction(f)

utils.save_results(pred.to('cpu').detach(), 'nn')
