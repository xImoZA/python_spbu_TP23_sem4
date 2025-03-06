import numpy as np
from numpy.typing import NDArray
from src.homeworks.homework2.KDTree import T
from src.homeworks.homework2.KNNClassifier import C


def train_test_split(
    X: NDArray[T], y: NDArray[C], test_size=0.2, shuffle=True
) -> tuple[NDArray[T], NDArray[T], NDArray[C], NDArray[C]]:
    n_samples = len(X)
    n_test = int(n_samples * test_size)

    i = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(i)

    test_i = i[:n_test]
    train_i = i[n_test:]

    X_train, X_test = X[train_i], X[test_i]
    y_train, y_test = y[train_i], y[test_i]

    return X_train, X_test, y_train, y_test
