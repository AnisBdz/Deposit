#!/usr/bin/python
# -*- coding: utf-8 -*-
import utils
import pandas as pd
import numpy as np

# paramètres
size = (5000, 1002)

# charger les données nécessaires
X_train, y_train, X_test, y_test = utils.load_data(size)
data = utils.load_test_data()

k = 2
d = X_train.shape[1]

# génerer la matrice weights
weights = np.random.normal(loc=0, scale=0.1, size=(d + 1, k))

# ajouter une première colonne de "1" à la matrice X des entrées
ones_train = np.ones(X_train.shape[0]).reshape(-1, 1)
ones_test  = np.ones(X_test.shape[0]).reshape(-1, 1)
X_train = np.concatenate((ones_train, X_train), axis=1)
X_test  = np.concatenate((ones_test,  X_test),  axis=1)

# softmax
def softmax(z):
    exp = np.exp(z - np.max(z, axis=1).reshape(-1, 1))
    sum = np.sum(exp, axis=1)
    return exp / sum[:,None]

# output
def output(X, weights):
    return softmax(np.dot(X, weights))

f = output(X_train, weights)
# print(f)

# prediction
def prediction(f):
    return np.argmax(f, axis=1)

p = prediction(f)
# print(p)

# one hot encoding
def one_hot_encode(y, k):
    return np.eye(k, k)[y.flatten().astype('int')].reshape((-1, k))

ohe = one_hot_encode(y_train, k)
# print(ohe)

# cross entropy
def crossentropy(f, y_one_hot):
    f = np.clip(f, 1e-7, 1.0 - 1e-7)
    N = f.shape[0]
    return (-1.0 / N) * np.sum(y_one_hot * np.log(f))

# c = crossentropy(f, ohe)
# print(c)

# gradient
def gradient(X, f, y_one_hot):
    return -np.dot(np.transpose(X), y_one_hot - f) / X.shape[0]

g = gradient(X_train, f, ohe)
# print(g)

# error rate
def error_rate(labels, preds):
    len_all = len(preds)
    faux    = preds != labels[:,0]
    return np.sum(faux) / float(len_all)

e = error_rate(y_train, p)
# print(e)

y_one_hot_train = one_hot_encode(y_train, k)
y_one_hot_test  = one_hot_encode(y_test,  k)

# print(f)
# print(crossentropy(f, y_train))

# entraînement
for i in range(10000):
    f_train = output(X_train, weights)
    weights = weights - 0.01 * gradient(X_train, f_train, y_one_hot_train)

    if i % 100 == 0:
        y_pred_train = prediction(f_train)
        error_train  = error_rate(y_train, y_pred_train)

        f_test      = output(X_test, weights)
        y_pred_test = prediction(f_test)
        error_test  = error_rate(y_test, y_pred_test)
        loss    = crossentropy(f_train, y_one_hot_train)
        print("iter : %-2d, loss : %.4f, error train : %.4f, error test : %.4f" % (i, loss.item(), error_train.item(), error_test.item()))


ones_data = np.ones(data.shape[0]).reshape(-1, 1)
data = np.concatenate((ones_data, data), axis=1)
f = output(data, weights)
predictions = prediction(f)
utils.save_results(predictions, 'lr')
