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
        self.clss: dict[Point[T], list[C]] | None = None
        self.all_classes: set[C] = set()

    def fit(self, X: NDArray[T], y: NDArray[C]):
        if X.shape[0] != y.shape[0]:
            raise ValueError("The lengths of 'X' and 'y' must be the same.")

        data = [Point(X[i]) for i in range(len(X))]
        self.classifier = KDTree(data, leaf_size=self.leaf_size)

        self.clss = {}
        for point, cls in zip(data, y):
            self.clss.setdefault(point, []).append(cls)
            self.all_classes.add(cls)

    def predict_proba(self, points: list[Point[T]]) -> dict[Point[T], dict[C, float]]:
        if self.classifier is None or self.clss is None:
            raise ValueError("Classifier has not been fitted yet.")

        dict_neighbors: dict[Point[T], list[Point[T]]] = self.classifier.query(
            points, self.k
        )

        dict_clss = {}
        for point in points:
            neighbors = dict_neighbors[point]

            neighbors_clss: list[C] = []
            for neighbor in neighbors:
                clss = self.clss[neighbor]
                for cls in clss:
                    neighbors_clss.append(cls)

            total = len(neighbors_clss)
            odds: dict[C, float] = {}
            for cls in self.all_classes:
                odds[cls] = neighbors_clss.count(cls) / total

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
