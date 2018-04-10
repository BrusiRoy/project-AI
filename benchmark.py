from sklearn.model_selection import train_test_split
import time

from sklearn import datasets, svm, metrics
from utils import get_MNIST
from sklearn import datasets, metrics
from support_vector_machine import support_vector_machine
from random_forest import random_forest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# MNIST benchmark
data, target = get_MNIST()
X_train, X_test, y_train, y_test = train_test_split(data / 255., target, train_size=60000, stratify=target, random_state=0)


def benchmark(classifier, dataset_name, X_train, X_test, y_train, y_test):

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

def MLP_benchmark(classifier, dataset_name, X_train, X_test, y_train, y_test):
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


# XXX: This is done in parallel
#rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0, n_jobs=-1)
#benchmark(rf, 'MNIST', X_train, X_test, y_train, y_test)

#svm = svm.SVC(gamma=0.001, random_state=0)
#benchmark(svm, 'MNIST', X_train, X_test, y_train, y_test)

#gb = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)
#benchmark(gb, 'MNIST', X_train, X_test, y_train, y_test)

"""
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
"""

#MLP_benchmark(mlp, 'MNIST', X_train, X_test, y_train, y_test)