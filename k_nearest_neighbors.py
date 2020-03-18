#!/usr/bin/python
# -*- coding: utf-8 -*-
import utils
import pandas as pd
import numpy as np

# parameters
# nombre d'echantillons
size = (6000, 3000)

# les différent valeus de k à tester
ks = range(3, 100)

# charger les données nécessaires
X_train, y_train, X_test, y_test = utils.load_data(size)

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

# calculer le taux d'erreur
def error_rate(predictions, labels):
    return len(labels[labels != predictions]) / float(len(labels))

# tester plusieurs parameters
for k in ks:

    # calculer les predictions
    predictions = []
    for x in utils.progress_bar(X_test):
        predictions.append(prediction(y_train[neighbors(X_train, x, k)]))

    # afficher les résultats
    print('k = %-3d  -->   error_rate = %.2f' % (k, error_rate(predictions, y_test.flatten())))
