from sklearn.model_selection import train_test_split
import time
import pandas as pd

from sklearn import datasets, svm, metrics
from sklearn.preprocessing import LabelEncoder
from utils import get_MNIST, get_bank, get_bank_full, get_bank_additional, get_bank_additional_full, get_abalone, get_glass, get_tic_tac_toe
from utils import create_GB, create_MLP, create_RF, create_SVM, create_XGB
from utils import create_optimized_GB, create_optimized_MLP, create_optimized_RF, create_optimized_SVM, create_optimized_SVM, create_optimized_XGB
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#result_df = pd.DataFrame(columns=['dataset', 'algo', 'accuracy', 'time'])


def benchmark(classifier, dataset_name, X_train, X_test, y_train, y_test):
    """Benchmark the classifier"""
    global result_df

    start_time = time.time()
    total_time = 0
    accuracy = 0

    if classifier.__class__.__name__ == 'MLPClassifier':
        # Fitting
        classifier.fit(X_train, y_train.values.ravel())

        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)

        # Get the time
        total_time = time.time() - start_time
        accuracy = test_score

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
                result_df.loc[len(result_df)] = [dataset_name, classifier.__class__.__name__, 0, 0]
                return
        # Fitting
        classifier.fit(X_train, y_train.values.ravel())

        # XXX: Maybe the new predicted is not good, might have to revert
        #predicted = classifier.predict(X_test)
        predicted = [round(value) for value in classifier.predict(X_test)]

        # Get the time
        total_time = time.time() - start_time
        accuracy = accuracy_score(y_test, predicted)

        # Print result to file
        with open(f"results/{classifier.__class__.__name__}/{dataset_name}-{start_time}.txt", 'a') as f:
            f.write(f'Result for the {classifier.__class__.__name__} on the {dataset_name} dataset with the following specification:\n')
            f.write(str(classifier))
            f.write('Classification Report:\n')
            f.write(str(metrics.classification_report(y_test, predicted)))
            f.write('Confusion matrix:\n')
            f.write(str(metrics.confusion_matrix(y_test, predicted)))
            f.write(f'\nExecution time: {total_time}')

    result_df.loc[len(result_df)] = [dataset_name, classifier.__class__.__name__, accuracy, total_time]


def print_fitted_model(classifier, dataset_name, X_test, y_test):
    global optimized_result_df

    start_time = time.time()
    predicted = [round(value) for value in classifier.predict(X_test)]
    total_time = time.time() - start_time
    accuracy = accuracy_score(y_test, predicted)
    optimized_result_df.loc[len(optimized_result_df)] = [dataset_name, classifier.__class__.__name__, accuracy, total_time]


"""
for dataset in [get_bank, get_bank_full, get_bank_additional, get_bank_additional_full, get_abalone, get_glass, get_tic_tac_toe, get_MNIST]:
    dataset_name, data, target = dataset()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)
    for classifier in [create_MLP, create_GB, create_RF, create_SVM, create_XGB]:
        print(f'Now running the {classifier.__name__} with the {dataset.__name__} dataset')
        benchmark(classifier(), dataset_name, X_train, X_test, y_train, y_test)
        print(result_df)

result_df.to_csv('results/all_results.csv')
"""
if __name__ == '__main__':
    optimized_result_df = pd.DataFrame(columns=['dataset', 'algo', 'accuracy', 'time'])
    for dataset in [get_bank]:
        dataset_name, data, target = dataset()
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)
        for classifier in [create_optimized_GB, create_optimized_MLP, create_optimized_RF, create_optimized_SVM, create_optimized_XGB]:
            print(f'Now running the {classifier.__name__} with the {dataset.__name__} dataset')
            trained_classifier = classifier(X_train, y_train)
            print_fitted_model(trained_classifier, dataset_name, X_test, y_test)
            print(optimized_result_df)
            with open(f"results/best_models_params.txt", 'a') as f:
                f.write(f'Here are the classifier specs:\n')
                f.write(str(trained_classifier) + '\n')

    with open('results/all_optimized_results.csv', 'a') as f:
        optimized_result_df.to_csv(f)
