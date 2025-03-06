from typing import Any, Generic

import numpy as np
from src.homeworks.homework2.KDTree import KDTree, Point, T


class KNNClassifier(Generic[T]):
    def __init__(self, k: int, leaf_size: int):
        self.k: int = k
        self.leaf_size: int = leaf_size
        self.classifier: KDTree[T] | None = None
        self.clss: dict[Point[T], Any] | None = None

    def fit(self, data: list[Point[T]], clss: list[Any]):
        self.classifier = KDTree(data, leaf_size=self.leaf_size)
        self.clss = {point: cls for point, cls in zip(data, clss)}

    def predict_proba(self, points: list[Point[T]]) -> dict[Point[T], dict[Any, float]]:
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

    def predict(self, points: list[Point[T]]) -> list[Any]:
        dict_clss: dict[Point[T], dict[Any, float]] = self.predict_proba(points)
        return [
            max(dict_clss[point].items(), key=lambda item: item[1])[0]
            for point in points
        ]
