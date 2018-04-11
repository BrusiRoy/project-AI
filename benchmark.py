from sklearn.model_selection import train_test_split
import time
import pandas as pd

from sklearn import datasets, svm, metrics
from sklearn.preprocessing import LabelEncoder
from utils import get_MNIST, get_bank, get_bank_full, get_bank_additional, get_bank_additional_full, get_abalone, get_glass, get_tic_tac_toe
from utils import create_GB, create_MLP, create_RF, create_SVM, create_XGB
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append("C:\\Users\\brusi\\Desktop\\xgboost\\python-package")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def benchmark(classifier, dataset_name, X_train, X_test, y_train, y_test):
    """Benchmark the classifier"""

    start_time = time.time()

    if classifier.__class__.__name__ == 'MLPClassifier':
        # Fitting
        classifier.fit(X_train, y_train.values.ravel())

        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)

        # Get the time
        total_time = time.time() - start_time

        # Print result to file
        with open(f"results/{classifier.__class__.__name__}/{dataset_name}-{start_time}.txt", 'a') as f:
            f.write(f'Result for the random forest on the {dataset_name} dataset with the following specification:\n')
            f.write(str(classifier))
            f.write(f'\nTrain score: {train_score}\n')
            f.write(f'Test score: {test_score}\n')
            f.write(f'\nExecution time: {total_time}')
    else:
        if classifier.__class__.__name__ == 'SVC' and len(X_train) > 8000:
            # If the classifier is a SVM and the dataset is larger than 8000, stop early...
            with open(f"results/{classifier.__class__.__name__}/{dataset_name}-{start_time}.txt", 'a') as f:
                f.write(f'Result for the {classifier.__class__.__name__} on the {dataset_name} dataset with the following specification:\n')
                f.write('Dataset too big for SVC!')
                return
        # Fitting
        classifier.fit(X_train, y_train.values.ravel())

        # XXX: Maybe the new predicted is not good, might have to revert
        #predicted = classifier.predict(X_test)
        predicted = [round(value) for value in classifier.predict(X_test)]

        # Get the time
        total_time = time.time() - start_time

        # Print result to file
        with open(f"results/{classifier.__class__.__name__}/{dataset_name}-{start_time}.txt", 'a') as f:
            f.write(f'Result for the {classifier.__class__.__name__} on the {dataset_name} dataset with the following specification:\n')
            f.write(str(classifier))
            f.write('Classification Report:\n')
            f.write(str(metrics.classification_report(y_test, predicted)))
            f.write('Confusion matrix:\n')
            f.write(str(metrics.confusion_matrix(y_test, predicted)))
            f.write(f'\nExecution time: {total_time}')


for dataset in [get_bank, get_bank_full, get_bank_additional, get_bank_additional_full, get_abalone, get_glass, get_tic_tac_toe, get_MNIST]:
    dataset_name, data, target = dataset()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)
    for classifier in [create_MLP, create_GB, create_RF, create_SVM, create_XGB]:
        print(f'Now running the {classifier.__name__} with the {dataset.__name__} dataset')
        benchmark(classifier(), dataset_name, X_train, X_test, y_train, y_test)
