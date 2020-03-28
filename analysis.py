#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, shutil
from scipy import stats

# clear output folders
shutil.rmtree('plots/columns')
os.mkdir('plots/columns')

# load data
train_data   = pd.read_csv('data/bank_train_data.csv')
train_labels = pd.read_csv('data/bank_train_labels.csv')
test_data    = pd.read_csv('data/bank_test_data.csv')

print(train_data.head())
print(test_data.head())

# split columns by type
numerical_columns   = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
categorical_columns = ['job', 'marital', 'education', 'contact', 'day', 'month', 'poutcome', 'default', 'housing', 'loan']

# show data distribution for each columns
print('Plotting numerical data distribution...')
for column in numerical_columns:
    print('- %s' % (column))
    plt.figure()
    plt.title(column)
    plt.hist(train_data[column], color='r')
    plt.hist(test_data[column], color='b')
    plt.savefig('plots/columns/%s.jpg' % (column))

# show data distribution for each columns
print('Plotting categorical data distribution...')
for column in categorical_columns:
    print('- %s' % (column))
    plt.figure()
    plt.title(column)
    train_data[column].value_counts().plot(color='r', kind='bar')
    test_data[column].value_counts().plot( color='b', kind='bar')
    plt.savefig('plots/columns/%s.jpg' % (column))
