from typing import Any

import pytest

from src.homeworks.homework2.KDTree import Point, T
from src.homeworks.homework2.KNNClassifier import KNNClassifier


class TestKNNClassifier:
    classifier = KNNClassifier(5, 3)
    data = [
        Point((1.2, 3.4, 0.5), "red"),
        Point((2.3, 1.1, 4.5), "green"),
        Point((0.8, 2.2, 3.3), "blue"),
        Point((4.5, 0.5, 1.2), "red"),
        Point((3.1, 2.8, 2.0), "green"),
        Point((1.0, 1.0, 1.0), "blue"),
        Point((5.0, 3.0, 0.1), "red"),
        Point((2.5, 2.5, 2.5), "green"),
        Point((0.2, 4.0, 1.5), "blue"),
        Point((3.7, 1.9, 0.8), "red"),
        Point((1.5, 3.0, 2.2), "green"),
        Point((0.5, 0.5, 4.0), "blue"),
        Point((4.0, 2.0, 1.0), "red"),
        Point((2.0, 4.0, 3.0), "green"),
        Point((1.8, 1.2, 2.8), "blue"),
        Point((3.0, 0.0, 2.5), "red"),
        Point((1.0, 2.0, 3.0), "green"),
        Point((0.0, 3.5, 1.0), "blue"),
        Point((2.2, 1.5, 0.7), "red"),
        Point((3.5, 2.5, 1.5), "green"),
    ]
    clss = ["red", "green", "blue"]
    classifier.fit(data, clss)

    @pytest.mark.parametrize(
        "points,expected_cls",
        [
            ([Point((2.0, 2.0, 2.0))], {Point((2.0, 2.0, 2.0)): [0.2, 0.6, 0.2]}),
            ([Point((3.0, 1.0, 1.0))], {Point((3.0, 1.0, 1.0)): [0.8, 0.2, 0.0]}),
            ([Point((0.5, 3.5, 2.5))], {Point((0.5, 3.5, 2.5)): [0, 0.4, 0.6]}),
            (
                [
                    Point((2.0, 2.0, 2.0)),
                    Point((3.0, 1.0, 1.0)),
                    Point((0.5, 3.5, 2.5)),
                ],
                {
                    Point((2.0, 2.0, 2.0)): [0.2, 0.6, 0.2],
                    Point((3.0, 1.0, 1.0)): [0.8, 0.2, 0.0],
                    Point((0.5, 3.5, 2.5)): [0, 0.4, 0.6],
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
            ([Point((2.0, 2.0, 2.0))], ["green"]),
            ([Point((3.0, 1.0, 1.0))], ["red"]),
            ([Point((0.5, 3.5, 2.5))], ["blue"]),
            (
                [
                    Point((2.0, 2.0, 2.0)),
                    Point((3.0, 1.0, 1.0)),
                    Point((0.5, 3.5, 2.5)),
                ],
                ["green", "red", "blue"],
            ),
        ],
    )
    def test_predict(self, points: list[Point[T]], expected_cls: list[Any]):
        assert self.classifier.predict(points) == expected_cls
