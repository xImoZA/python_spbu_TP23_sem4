from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from src.homeworks.homework2.KDTree import KDTree, Point, T

C = TypeVar("C", bound=np.generic)


class KNNClassifier(Generic[T, C]):
    def __init__(self, k: int, leaf_size: int):
        self.k: int = k
        self.leaf_size: int = leaf_size
        self.classifier: KDTree[T] | None = None
        self.clss: dict[Point[T], C] | None = None

    def fit(self, X: NDArray[T], y: NDArray[C]):
        if X.shape[0] != y.shape[0]:
            raise ValueError("The lengths of 'X' and 'y' must be the same.")

        data = [Point(X[i]) for i in range(len(X))]
        self.classifier = KDTree(data, leaf_size=self.leaf_size)
        self.clss = {point: cls for point, cls in zip(data, y)}

    def predict_proba(self, points: list[Point[T]]) -> dict[Point[T], dict[C, float]]:
        if self.classifier is None or self.clss is None:
            raise ValueError("Classifier has not been fitted yet.")

        dict_neighbors: dict[Point[T], list[Point[T]]] = self.classifier.query(
            points, self.k
        )

        dict_clss = {}
        for point in points:
            neighbors = dict_neighbors[point]
            odds = {}
            for cls in set(self.clss.values()):
                odds[cls] = (
                    len([point for point in neighbors if self.clss[point] == cls])
                    / self.k
                )

            dict_clss[point] = odds

        return dict_clss

    def predict(self, data: NDArray[T]) -> NDArray[C]:
        points = [Point(point) for point in data]
        dict_clss: dict[Point[T], dict[C, float]] = self.predict_proba(points)

        return np.array(
            [
                max(dict_clss[point].items(), key=lambda item: item[1])[0]
                for point in points
            ]
        )
