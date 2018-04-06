from sklearn.model_selection import train_test_split

from utils import get_MNIST
from support_vector_machine import support_vector_machine

# MNIST benchmark
data, target = get_MNIST()
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, stratify=target, random_state=0)
support_vector_machine('MNIST', X_train, X_test, y_train, y_test)
