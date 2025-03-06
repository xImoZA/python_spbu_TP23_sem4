from typing import Any, Generic

import numpy as np
import pytest
from src.homeworks.homework2.KDTree import Point, T
from src.homeworks.homework2.KNNClassifier import KNNClassifier


class TestKNNClassifier(Generic[T]):
    classifier: KNNClassifier[T] = KNNClassifier(5, 3)
    data: list[Point[T]] = [
        Point(np.array([1.2, 3.4, 0.5])),
        Point(np.array([2.3, 1.1, 4.5])),
        Point(np.array([0.8, 2.2, 3.3])),
        Point(np.array([4.5, 0.5, 1.2])),
        Point(np.array([3.1, 2.8, 2.0])),
        Point(np.array([1.0, 1.0, 1.0])),
        Point(np.array([5.0, 3.0, 0.1])),
        Point(np.array([2.5, 2.5, 2.5])),
        Point(np.array([0.2, 4.0, 1.5])),
        Point(np.array([3.7, 1.9, 0.8])),
        Point(np.array([1.5, 3.0, 2.2])),
        Point(np.array([0.5, 0.5, 4.0])),
        Point(np.array([4.0, 2.0, 1.0])),
        Point(np.array([2.0, 4.0, 3.0])),
        Point(np.array([1.8, 1.2, 2.8])),
        Point(np.array([3.0, 0.0, 2.5])),
        Point(np.array([1.0, 2.0, 3.0])),
        Point(np.array([0.0, 3.5, 1.0])),
        Point(np.array([2.2, 1.5, 0.7])),
        Point(np.array([3.5, 2.5, 1.5])),
    ]
    clss = [
        "red",
        "green",
        "blue",
        "red",
        "green",
        "blue",
        "red",
        "green",
        "blue",
        "red",
        "green",
        "blue",
        "red",
        "green",
        "blue",
        "red",
        "green",
        "blue",
        "red",
        "green",
    ]
    classifier.fit(data, clss)

    @pytest.mark.parametrize(
        "points,expected_cls",
        [
            (
                [Point(np.array([2.0, 2.0, 2.0]))],
                {
                    Point(np.array([2.0, 2.0, 2.0])): {
                        "red": 0.2,
                        "green": 0.6,
                        "blue": 0.2,
                    }
                },
            ),
            (
                [Point(np.array([3.0, 1.0, 1.0]))],
                {
                    Point(np.array([3.0, 1.0, 1.0])): {
                        "red": 0.8,
                        "green": 0.2,
                        "blue": 0.0,
                    }
                },
            ),
            (
                [Point(np.array([0.5, 3.5, 2.5]))],
                {
                    Point(np.array([0.5, 3.5, 2.5])): {
                        "red": 0,
                        "green": 0.4,
                        "blue": 0.6,
                    }
                },
            ),
            (
                [
                    Point(np.array([2.0, 2.0, 2.0])),
                    Point(np.array([3.0, 1.0, 1.0])),
                    Point(np.array([0.5, 3.5, 2.5])),
                ],
                {
                    Point(np.array([2.0, 2.0, 2.0])): {
                        "red": 0.2,
                        "green": 0.6,
                        "blue": 0.2,
                    },
                    Point(np.array([3.0, 1.0, 1.0])): {
                        "red": 0.8,
                        "green": 0.2,
                        "blue": 0.0,
                    },
                    Point(np.array([0.5, 3.5, 2.5])): {
                        "red": 0,
                        "green": 0.4,
                        "blue": 0.6,
                    },
                },
            ),
        ],
    )
    def test_predict_proba(
        self, points: list[Point[T]], expected_cls: dict[Point[T], list[Point[T]]]
    ):
        actual = self.classifier.predict_proba(points)

        for point in points:
            assert actual[point] == expected_cls[point]

    @pytest.mark.parametrize(
        "points,expected_cls",
        [
            ([Point(np.array([2.0, 2.0, 2.0]))], ["green"]),
            ([Point(np.array([3.0, 1.0, 1.0]))], ["red"]),
            ([Point(np.array([0.5, 3.5, 2.5]))], ["blue"]),
            (
                [
                    Point(np.array([2.0, 2.0, 2.0])),
                    Point(np.array([3.0, 1.0, 1.0])),
                    Point(np.array([0.5, 3.5, 2.5])),
                ],
                ["green", "red", "blue"],
            ),
        ],
    )
    def test_predict(self, points: list[Point[T]], expected_cls: list[Any]):
        assert self.classifier.predict(points) == expected_cls
