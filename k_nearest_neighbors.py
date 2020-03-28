#!/usr/bin/python
# -*- coding: utf-8 -*-
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# calculer la distance euclidienne
def euclidian_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

# rechercher les k proches voisins
def neighbors(data, x, k):
    # calculer les distances
    distances = np.array(map(lambda row: euclidian_distance(x, row), data))

    # trie des données selon les distances
    # retourner les indices
    return np.argsort(distances)[:k]

# predire la classe selon le voisinage
def prediction(voisinage):
    # calculer la fréquence des labels unique
    (unique, counts) = np.unique(voisinage, return_counts=True)

    # retouner la prediction (la classe avec le plus nombre d'occurence)
    return unique[np.argmax(counts)]

def predictions(X_test, X_train, y_train, k):
    predictions = []
    for x in utils.progress_bar(X_test):
        predictions.append(prediction(y_train[neighbors(X_train, x, k)]))

    return np.array(predictions)

# calculer le taux d'erreur
def error_rate(predictions, labels):
    return len(labels[labels != predictions]) / float(len(labels))


# tester des differenets tailles d'echantillons
# error_rates = []

# for i in range(1, 10):
#     print('i: %d' % (i))
#     X_train, y_train, X_test, y_test = utils.load_data((i * 700, i * 300))
#
#     pred = predictions(X_test, X_train, y_train, 10)
#     faux_positifs = pred[(pred == 1) & (y_test.flatten() == 0)]
#     faux_negatifs = pred[(pred == 0) & (y_test.flatten() == 1)]
#
#     rate = error_rate(pred, y_test.flatten())
#     error_rates.append(rate)
#
#     print("Error rate: %f" % (rate))
#     print("Nombre de faux positifs: %d" % (len(faux_positifs)))
#     print("Nombre de faux negatifs: %d" % (len(faux_negatifs)))


# plot
# plt.title('error rates')
# plt.plot(np.arange(9) * 1000, error_rates)
# plt.savefig('plots/knn_error_rates.jpg')

# generate test labels
# charger les données nécessaires
X_train, y_train, X_test, y_test = utils.load_data((9200, 1))
data = utils.load_test_data()
utils.save_results(predictions(data, X_train, y_train, 10), 'knn')
