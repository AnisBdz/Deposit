j,$;#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np

# chargement les données en dataframe
size = 1000

result_data = pd.get_dummies(pd.read_csv('data/bank_test_data.csv')).values

data   = pd.get_dummies(pd.read_csv('data/bank_train_data.csv')).head(size).values
labels = pd.get_dummies(pd.read_csv('data/bank_train_labels.csv')).head(size).values

p70 = int(data.shape[0]*0.7)
train_data = data[:p70]
test_data  = data[p70:]

train_labels = labels[:p70]
test_labels  = labels[p70:]

# conversion de données en tensor
train_X = torch.tensor(train_data).float()
train_y = torch.tensor(train_labels).float()
test_X  = torch.tensor(test_data).float()
test_y  = torch.tensor(test_labels).float()
=^ù)
result_X = torch.tensor(result_data).float()

d = train_data.shape[1]

# définition du modèle
class Modele(torch.nn.Module):
    def __init__(self, d):
        super(Modele, self).__init__()

        _2d = d + 10
        _3d = d + 50

        self.l1 = torch.nn.Linear(d,  _2d)
        self.l2 = torch.nn.Linear(_2d, _3d)
        self.l3 = torch.nn.Linear(_3d, _2d)
        self.l4 = torch.nn.Linear(_2d, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1   = self.sigmoid(self.l1(x))
        out2   = self.sigmoid(self.l2(out1))
        out3   = self.sigmoid(self.l3(out2))
        pred_y = self.sigmoid(self.l4(out3))
        return pred_y

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

# critère de Loss
criterion = torch.nn.BCELoss()

# optimizer
optimizer = torch.optim.SGD(modele.parameters(), lr=0.01)

# entrainemnt
for epoch in range(100000):
    f_train = modele(train_X)
    loss = criterion(f_train, train_y)

    if epoch % 100 == 0:
        pred_train  = prediction(f_train)
        error_train = error_rate(pred_train, train_y)

        f_test      = modele(test_X)
        pred_test   = prediction(f_test)
        error_test  = error_rate(pred_test, test_y)


        print('epoch: ', epoch, 'loss: ', loss.item(), 'error_train', error_train.item(), 'error_test', error_test.item())

        if (error_test.item() < 0.257 and error_train.item() < 0.309):
            f_result      = modele(result_X)
            pred_result   = prediction(f_result)
            np.savetxt('bank_test_results.csv', pred_result.detach().numpy())
            break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
