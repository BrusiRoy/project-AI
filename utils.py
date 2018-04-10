from sklearn.datasets import fetch_mldata
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_MNIST():
    """Returns a (name, data, target) tuple of the MNIST dataset (70 000 items)"""
    mnist = fetch_mldata('MNIST original', data_home='./data/')
    return ('MNIST', mnist.data, mnist.target)


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
