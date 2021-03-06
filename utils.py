#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas
import numpy
import unittest
import re
from time import time, sleep
from os import path, mkdir, remove
from shutil import rmtree
from tqdm import tqdm

numpy.random.seed(100000)

# charger les données nécessaires pour l'entrainement des modèles
# en prenant des échantillon aléatoires
def load_data(size):
    train_size, test_size = size

    if train_size <= 0 or test_size <= 0:
        raise Exception('train and test data size must be positive integers, (%d, %d) given.' % size)

    # chargement des données
    data   = pandas.get_dummies(pandas.read_csv('data/bank_train_data.csv')).values
    labels = pandas.read_csv('data/bank_train_labels.csv').values.astype('int')

    # mélange des données
    all = numpy.concatenate((data, labels), axis = 1)
    numpy.random.shuffle(all)

    labels = all[:,-1]

    has_one  = labels == 1
    has_zero = labels == 0

    with_ones  = all[has_one]
    with_zeros = all[has_zero]

    train_size = int(train_size / 2)
    test_size  = int(test_size / 2)

    # extrait des données d'entrainemnt et de test
    train = numpy.concatenate((with_ones[:train_size], with_zeros[:train_size]))
    test  = numpy.concatenate((with_ones[train_size : train_size + test_size], with_zeros[train_size : train_size + test_size]))

    X_train = train[:,:-1]
    y_train = train[:,-1:]

    X_test = test[:,:-1]
    y_test = test[:,-1:]

    return (X_train, y_train, X_test, y_test)

# charger les données nécessaires pour générer les résultats finales
def load_test_data():
    data   = pandas.get_dummies(pandas.read_csv('data/bank_test_data.csv')).values
    return data

# sauvgarder les résultats
def save_results(results, method):
    if re.match("[a-z]+", method) == None:
        raise Exception('method should be all lowercase alphabetic string')

    timestamp = int(time())
    name = '%d_%s' % (timestamp, method)
    mkdir('out/%s' % (name))
    numpy.savetxt('out/%s/bank_test_results.csv' % (name), results)

    return name

# barre de progression
def progress_bar(l):
    return tqdm(l, leave=False, dynamic_ncols=True)

# tester le bon fonctionnement des méthodes
class TestUtils(unittest.TestCase):

    def test_load_data(self):
        with self.assertRaises(Exception) as context:
            load_data(size = (-10, -10))

        self.assertTrue('train and test data size must be positive integers' in str(context.exception))

        X_train, y_train, X_test, y_test = load_data(size = (100, 50))

        self.assertEqual(X_train.shape, (100, 51))
        self.assertEqual(y_train.shape, (100, 1))

        self.assertEqual(X_test.shape, (50, 51))
        self.assertEqual(y_test.shape, (50, 1))

    def test_load_test_data(self):
        data = load_test_data()
        self.assertEqual(data.shape, (1002, 51))

    def test_save_results(self):
        results = numpy.ones(1002)
        folder = ''

        with self.assertRaises(Exception) as context:
            save_results(results, "CNN")
        self.assertTrue('method should be all lowercase alphabetic string' in str(context.exception))

        try:
            name = save_results(results, "cnn")
            self.assertNotEqual(re.match("\d{10}_[a-z]+", name), None)

            folder = 'out/%s/' % (name)
            filename = '%s/bank_test_results.csv' % (folder)
            self.assertTrue(path.exists(filename))

            results = numpy.loadtxt(filename)
            self.assertEqual(results.shape[0], 1002)

        finally:
            if path.exists(folder):
                rmtree(folder)

    def test_progress_bar(self):
        for i in progress_bar(range(1000)):
            pass

# créer les dossier s'ils n'existent pas
if not path.exists('out'):
    mkdir('out')

if __name__ == '__main__':
    unittest.main()
