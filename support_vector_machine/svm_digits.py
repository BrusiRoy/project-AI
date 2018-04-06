import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# The digits dataset
digits = datasets.load_digits() 

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create the train and test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, stratify=digits.target, random_state=0)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# Fitting
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
expected = y_test
predicted = classifier.predict(X_test)

print('Result for the Support Vector Machine implementation with the following specification:')
print(classifier)
print('Classification Report:')
print(metrics.classification_report(expected, predicted))
print('Confusion matrix:')
print(metrics.confusion_matrix(expected, predicted))