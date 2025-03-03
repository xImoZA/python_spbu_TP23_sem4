from typing import Any

from src.homeworks.homework2.KDTree import KDTree, Point, T


class KNNClassifier:
    def __init__(self, k: int, leaf_size: int):
        self.k: int = k
        self.leaf_size: int = leaf_size
        self.classifier: KDTree | None = None
        self.clss: list[Any] | None = None

    def fit(self, data: list[Point[T]], clss: list[Any]):
        self.classifier = KDTree(data, leaf_size=3)
        self.clss = clss

    def predict_proba(self, points: list[Point[T]]) -> dict[Point[T], list[float]]:
        if self.classifier is None or self.clss is None:
            raise ValueError("Classifier has not been fitted yet.")

        dict_neighbors: dict[Point[T], list[Point[T]]] = self.classifier.query(
            points, self.k
        )

        dict_clss = {}
        for point in points:
            neighbors = dict_neighbors[point]
            odds = [
                len([p for p in neighbors if p.cls == cls]) / self.k
                for cls in self.clss
            ]

            dict_clss[point] = odds

        return dict_clss

    def get_cls(self, odds: list[float]) -> Any:
        if self.clss is None:
            raise ValueError("Classifier has not been fitted yet.")

        ind = odds.index(max(odds))
        return self.clss[ind]

    def predict(self, points: list[Point[T]]) -> list[float]:
        dict_clss: dict[Point[T], list[float]] = self.predict_proba(points)
        return [self.get_cls(odds) for point, odds in dict_clss.items()]
