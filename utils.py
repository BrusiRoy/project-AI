from sklearn.datasets import fetch_mldata
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, svm, metrics
import sys
sys.path.append("C:\\Users\\brusi\\Desktop\\xgboost\\python-package")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def get_MNIST():
    """Returns a (name, data, target) tuple of the MNIST dataset (70 000 items)"""
    mnist = fetch_mldata('MNIST original', data_home='./data/')
    return ('MNIST', pd.DataFrame(mnist.data / 255.), pd.DataFrame(mnist.target))


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


def create_optimized_RF(X_train, y_train):
    """Returns an optimized RandomForestClassifier"""
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3, 4],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split': [4, 6, 8, 10],
        'n_estimators': [100, 200, 300, 1000]
    }

    # Create the base model
    classifier = RandomForestClassifier()

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train.values.ravel())
    return grid_search.best_estimator_


def create_SVM():
    """Retruns a svm.SVC"""
    return svm.SVC(gamma=0.001)


def create_optimized_SVM(X_train, y_train):
    """Retruns an optimized svm.SVC"""
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1, 1]
    }

    # Create the base model
    classifier = svm.SVC()

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train.values.ravel())
    return grid_search.best_estimator_


def create_GB():
    """Returns a Gradient boosting"""
    return GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)


def create_optimized_GB(X_train, y_train):
    """Returns an optimized Gradient boosting"""
    param_grid = {
        'loss': ['deviance'],
        'learning_rate': [0.05, 0.1, 1],
        'min_samples_leaf': [2, 4, 5],
        'min_samples_split': [4, 6, 8, 10],
        'max_depth': [3, 5, 8],
        'subsample': [0.5, 1.0],
        'n_estimators': [10, 100, 200]
    }

    # Create the base model
    classifier = GradientBoostingClassifier()

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train.values.ravel())
    return grid_search.best_estimator_


def create_MLP():
    """Returns a MLP"""
    return MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)


def create_optimized_MLP(X_train, y_train):
    """Returns an optimized MLP"""
    param_grid = {
        'solver': ['sgd', 'adam'],
        'hidden_layer_sizes': [(7, 7), (128,), (128, 7)],
        'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'learning_rate_init': [0.01, 0.05, 0.1, 0.2, 1],
        'epsilon': [1e-3, 1e-7, 1e-8, 1e-9]
    }

    # Create the base model
    classifier = MLPClassifier()

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train.values.ravel())
    return grid_search.best_estimator_


def create_XGB():
    """Returns a XGB classifier"""
    return XGBClassifier()


def create_optimized_XGB(X_train, y_train):
    """Returns an optimized XGB classifier"""
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 1],
        'max_depth': [3, 5, 8],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators': [10, 100, 200]
    }

    # Create the base model
    classifier = XGBClassifier()

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train.values.ravel())
    return grid_search.best_estimator_


def create_bank_tuned_GB(X_train, y_train):
    return GradientBoostingClassifier(criterion='friedman_mse', init=None,
        learning_rate=0.1, loss='deviance', max_depth=8,
        max_features='sqrt', max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=2, min_samples_split=6,
        min_weight_fraction_leaf=0.0, n_estimators=100,
        presort='auto', random_state=None, subsample=1.0, verbose=0,
        warm_start=False).fit(X_train, y_train.values.ravel())


def create_bank_tuned_MLP(X_train, y_train):
    return MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
        beta_2=0.999, early_stopping=False, epsilon=1e-07,
        hidden_layer_sizes=(128,), learning_rate='constant',
        learning_rate_init=0.1, max_iter=200, momentum=0.9,
        nesterovs_momentum=True, power_t=0.5, random_state=None,
        shuffle=True, solver='adam', tol=0.01, validation_fraction=0.1,
        verbose=False, warm_start=False).fit(X_train, y_train.values.ravel())


def create_bank_tuned_RF(X_train, y_train):
    return RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        max_depth=80, max_features=4, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=2, min_samples_split=6,
        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
        oob_score=False, random_state=None, verbose=0,
        warm_start=False).fit(X_train, y_train.values.ravel())


def create_bank_tuned_SVM(X_train, y_train):
    return svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False).fit(X_train, y_train.values.ravel())


def create_bank_tuned_XGB(X_train, y_train):
    return XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        colsample_bytree=1.0, gamma=5, learning_rate=0.05, max_delta_step=0,
        max_depth=8, min_child_weight=1, missing=None, n_estimators=200,
        n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
        silent=True, subsample=0.6).fit(X_train, y_train.values.ravel())