import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# The iris dataset
iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Create the train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target, test_size=0.5, stratify=iris.target, random_state=0)


def random_forest(dataset_name, X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)

    start_time = time.time()
    # Fitting
    classifier.fit(X_train, y_train)

    expected = y_test
    predicted = classifier.predict(X_test)

    total_time = time.time() - start_time

    with open(f"results/random_forest/rf-{dataset_name}-{start_time}.txt", 'a') as f:
        f.write(f'Result for the random forest on the {dataset_name} dataset with the following specification:\n')
        f.write(str(classifier))
        f.write('Classification Report:\n')
        f.write(str(metrics.classification_report(expected, predicted)))
        f.write('Confusion matrix:\n')
        f.write(str(metrics.confusion_matrix(expected, predicted)))
        f.write(f'\nExecution time: {total_time}')


random_forest('iris', X_train, X_test, y_train, y_test)