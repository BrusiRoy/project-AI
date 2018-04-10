from sklearn.datasets import fetch_mldata
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_MNIST():
    """Returns a (data, target) tuple of the MNIST dataset (70 000 items)"""
    mnist = fetch_mldata('MNIST original', data_home='./data/')
    return (mnist.data, mnist.target)


def get_bank():
    """Returns a (data, target) tuple of the Bank dataset (4521 items)"""
    df = pd.read_csv('data/bank/bank.csv', sep=';')

    for column in df:
        enc = LabelEncoder()
        enc.fit(df[column])
        df[column] = enc.transform(df[column])

    data = df.loc[:, df.columns != 'y']
    target = df.loc[:, df.columns == 'y']
    return (data, target)


def get_bank_full():
    """Returns a (data, target) tuple of the Bank-full dataset (45211 items)"""
    df = pd.read_csv('data/bank/bank-full.csv', sep=';')

    for column in df:
        enc = LabelEncoder()
        enc.fit(df[column])
        df[column] = enc.transform(df[column])

    data = df.loc[:, df.columns != 'y']
    target = df.loc[:, df.columns == 'y']
    return (data, target)
