from sklearn.datasets import fetch_mldata
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, svm, metrics


def get_MNIST():
    """Returns a (name, data, target) tuple of the MNIST dataset (70 000 items)"""
    mnist = fetch_mldata('MNIST original', data_home='./data/')
    return ('MNIST', mnist.data / 255., mnist.target)


def get_bank():
    """Returns a (name, data, target) tuple of the Bank dataset (4521 items)"""
    df = pd.read_csv('data/bank/bank.csv', sep=';')

    for column in df:
        enc = LabelEncoder()
        enc.fit(df[column])
        df[column] = enc.transform(df[column])

    data = df.loc[:, df.columns != 'y']
    target = df.loc[:, df.columns == 'y']
    return ('bank', data, target)


def get_bank_full():
    """Returns a (name, data, target) tuple of the Bank-full dataset (45211 items)"""
    df = pd.read_csv('data/bank/bank-full.csv', sep=';')

    for column in df:
        enc = LabelEncoder()
        enc.fit(df[column])
        df[column] = enc.transform(df[column])

    data = df.loc[:, df.columns != 'y']
    target = df.loc[:, df.columns == 'y']
    return ('bank-full', data, target)


def get_bank_additional():
    """Returns a (name, data, target) tuple of the bank-additional dataset (4118 items)"""
    df = pd.read_csv('data/bank-additional/bank-additional.csv', sep=';')

    for column in df:
        enc = LabelEncoder()
        enc.fit(df[column])
        df[column] = enc.transform(df[column])

    data = df.loc[:, df.columns != 'y']
    target = df.loc[:, df.columns == 'y']
    return ('bank-additional', data, target)


def get_bank_additional_full():
    """Returns a (name, data, target) tuple of the bank-additional-full dataset (41188 items)"""
    df = pd.read_csv('data/bank-additional/bank-additional-full.csv', sep=';')

    for column in df:
        enc = LabelEncoder()
        enc.fit(df[column])
        df[column] = enc.transform(df[column])

    data = df.loc[:, df.columns != 'y']
    target = df.loc[:, df.columns == 'y']
    return ('bank-additional-full', data, target)


def get_abalone():
    """Returns a (name, data, target) tuple of the Abalone dataset (4177 items)"""
    df = pd.read_csv('data/abalone/abalone.data', header=None)

    enc = LabelEncoder()
    enc.fit(df[df.columns[0]])
    df[df.columns[0]] = enc.transform(df[df.columns[0]])
    data = df[df.columns[:8]]
    target = df[df.columns[8:]]
    return ('Abalone', data, target)


def get_glass():
    """Returns a (name, data, target) tuple of the Glass dataset (214 items)"""
    df = pd.read_csv('data/glass/glass.data', header=None)

    data = df[df.columns[:10]]
    target = df[df.columns[10:]]
    return ('Glass', data, target)


def get_tic_tac_toe():
    """Returns a (name, data, target) tuple of the Tic Tac Toe dataset (958 items)"""
    df = pd.read_csv('data/tic-tac-toe/tic-tac-toe.data', header=None)

    for column in df:
        enc = LabelEncoder()
        enc.fit(df[column])
        df[column] = enc.transform(df[column])

    data = df[df.columns[:9]]
    target = df[df.columns[9:]]
    return ('Tic-Tac-Toe', data, target)


def create_RF():
    """Returns a RandomForestClassifier"""
    return RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)


def create_SVM():
    """Retruns a svm.SVC"""
    return svm.SVC(gamma=0.001)


def create_GB():
    """Returns a Gradient boosting"""
    return GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)


def create_MLP():
    """Returns a MLP"""
    return MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)