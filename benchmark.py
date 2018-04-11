from sklearn.model_selection import train_test_split
import time
import pandas as pd

from sklearn import datasets, svm, metrics
from sklearn.preprocessing import LabelEncoder
from utils import get_MNIST, get_bank, get_bank_full, get_bank_additional, get_bank_additional_full, get_abalone, get_glass, get_tic_tac_toe
from utils import create_GB, create_MLP, create_RF, create_SVM
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def benchmark(classifier, dataset_name, X_train, X_test, y_train, y_test):
    """Benchmark the classifier"""

    if classifier.__class__.__name__ == 'MLPClassifier':
        start_time = time.time()
        # Fitting
        classifier.fit(X_train, y_train)

        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)

        total_time = time.time() - start_time

        with open(f"results/{classifier.__class__.__name__}/{dataset_name}-{start_time}.txt", 'a') as f:
            f.write(f'Result for the random forest on the {dataset_name} dataset with the following specification:\n')
            f.write(str(classifier))
            f.write(f'\nTrain score: {train_score}\n')
            f.write(f'Test score: {test_score}\n')
            f.write(f'\nExecution time: {total_time}')
    else:
        if classifier.__class__.__name__ == 'SVC' and len(X_train > 8000):
            with open(f"results/{classifier.__class__.__name__}/{dataset_name}-{start_time}.txt", 'a') as f:
                f.write(f'Result for the {classifier.__class__.__name__} on the {dataset_name} dataset with the following specification:\n')
                f.write('Dataset too big for SVC!')
                return

        start_time = time.time()
        # Fitting
        classifier.fit(X_train, y_train)

        expected = y_test
        predicted = classifier.predict(X_test)

        total_time = time.time() - start_time

        with open(f"results/{classifier.__class__.__name__}/{dataset_name}-{start_time}.txt", 'a') as f:
            f.write(f'Result for the {classifier.__class__.__name__} on the {dataset_name} dataset with the following specification:\n')
            f.write(str(classifier))
            f.write('Classification Report:\n')
            f.write(str(metrics.classification_report(expected, predicted)))
            f.write('Confusion matrix:\n')
            f.write(str(metrics.confusion_matrix(expected, predicted)))
            f.write(f'\nExecution time: {total_time}')


for dataset in [get_MNIST, get_bank, get_bank_full, get_bank_additional, get_bank_additional_full, get_abalone, get_glass, get_tic_tac_toe]:
    dataset_name, data, target = dataset()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)
    for classifier in [create_GB, create_MLP, create_RF, create_SVM]:
        print(f'Now running the {classifier.__name__} with the {dataset.__name__} dataset')
        #benchmark(classifier(), dataset_name, X_train, X_test, y_train, y_test)
