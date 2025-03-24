from typing import Generic

import numpy as np
from numpy.typing import NDArray
from src.homeworks.homework2.KDTree import T
from src.homeworks.homework2.KNNClassifier import C


class MetricCalculator(Generic[T]):
    def __init__(self, y_pred: NDArray[C], y_true: NDArray[C]):
        if len(y_pred) != len(y_true):
            raise ValueError(
                "The lengths of lists 'y_pred' and 'y_true' must be the same."
            )

        self.y_pred: NDArray[C] = y_pred
        self.y_true: NDArray[C] = y_true

    def true_positive(self) -> int:
        return int(np.sum((self.y_pred == 1) & (self.y_true == 1)))

    def false_positive(self) -> int:
        return int(np.sum((self.y_pred == 1) & (self.y_true != 1)))

    def false_negative(self) -> int:
        return int(np.sum((self.y_pred != 1) & (self.y_true == 1)))

    def true_negative(self) -> int:
        return int(np.sum((self.y_pred != 1) & (self.y_true != 1)))

    def precision(self) -> float:
        tp = self.true_positive()
        fp = self.false_positive()

        return tp / (tp + fp)

    def recall(self) -> float:
        tp = self.true_positive()
        fn = self.false_negative()

        return tp / (tp + fn)

    def f1_score(self) -> float:
        precision = self.precision()
        recall = self.recall()

        return (2 * precision * recall) / (precision + recall)

    def accuracy(self) -> float:
        tp = self.true_positive()
        tn = self.true_negative()
        fp = self.false_positive()
        fn = self.false_negative()

        return (tp + tn) / (tp + tn + fp + fn)
