from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

import time

# The digits dataset
digits = datasets.load_digits() 

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
target = digits.target

# Create the train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, stratify=target, random_state=0)


def support_vector_machine(dataset_name, X_train, X_test, y_train, y_test):
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    start_time = time.time()
    # Fitting
    classifier.fit(X_train, y_train)

    expected = y_test
    predicted = classifier.predict(X_test)

    total_time = time.time() - start_time

    with open(f"results/support_vector_machine/svm-{dataset_name}-{start_time}.txt", 'a') as f:
        f.write(f'Result for the SVM on the {dataset_name} dataset with the following specification:\n')
        f.write(str(classifier))
        f.write('Classification Report:\n')
        f.write(str(metrics.classification_report(expected, predicted)))
        f.write('Confusion matrix:\n')
        f.write(str(metrics.confusion_matrix(expected, predicted)))
        f.write(f'\nExecution time: {total_time}')


support_vector_machine('digits', X_train, X_test, y_train, y_test)
