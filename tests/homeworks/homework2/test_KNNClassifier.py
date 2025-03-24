from typing import Generic

import numpy as np
import pytest
from numpy.typing import NDArray
from src.homeworks.homework2.KDTree import Point, T
from src.homeworks.homework2.KNNClassifier import C, KNNClassifier


class TestKNNClassifier(Generic[T, C]):
    classifier: KNNClassifier[T, C] = KNNClassifier(5, 3)
    data: NDArray[T] = np.array(
        [
            np.array([1.2, 3.4, 0.5]),
            np.array([2.3, 1.1, 4.5]),
            np.array([0.8, 2.2, 3.3]),
            np.array([4.5, 0.5, 1.2]),
            np.array([3.1, 2.8, 2.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([5.0, 3.0, 0.1]),
            np.array([2.5, 2.5, 2.5]),
            np.array([0.2, 4.0, 1.5]),
            np.array([3.7, 1.9, 0.8]),
            np.array([1.5, 3.0, 2.2]),
            np.array([0.5, 0.5, 4.0]),
            np.array([4.0, 2.0, 1.0]),
            np.array([2.0, 4.0, 3.0]),
            np.array([1.8, 1.2, 2.8]),
            np.array([3.0, 0.0, 2.5]),
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 3.5, 1.0]),
            np.array([2.2, 1.5, 0.7]),
            np.array([3.5, 2.5, 1.5]),
        ]
    )
    clss: NDArray[C] = np.array(
        [
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
    )
    classifier.fit(data, clss)

    @pytest.mark.parametrize(
        "points,expected_cls",
        [
            (
                [Point(np.array([2.0, 2.0, 2.0]))],
                {
                    Point(np.array([2.0, 2.0, 2.0])): {
                        np.str_("red"): 0.2,
                        np.str_("green"): 0.6,
                        np.str_("blue"): 0.2,
                    }
                },
            ),
            (
                [Point(np.array([3.0, 1.0, 1.0]))],
                {
                    Point(np.array([3.0, 1.0, 1.0])): {
                        np.str_("red"): 0.8,
                        np.str_("green"): 0.2,
                        np.str_("blue"): 0.0,
                    }
                },
            ),
            (
                [Point(np.array([0.5, 3.5, 2.5]))],
                {
                    Point(np.array([0.5, 3.5, 2.5])): {
                        np.str_("red"): 0,
                        np.str_("green"): 0.6,
                        np.str_("blue"): 0.4,
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
                        np.str_("red"): 0.2,
                        np.str_("green"): 0.6,
                        np.str_("blue"): 0.2,
                    },
                    Point(np.array([3.0, 1.0, 1.0])): {
                        np.str_("red"): 0.8,
                        np.str_("green"): 0.2,
                        np.str_("blue"): 0.0,
                    },
                    Point(np.array([0.5, 3.5, 2.5])): {
                        np.str_("red"): 0,
                        np.str_("green"): 0.6,
                        np.str_("blue"): 0.4,
                    },
                },
            ),
        ],
    )
    def test_predict_proba(
        self, points: list[Point[T]], expected_cls: dict[Point[T], dict[C, float]]
    ):
        actual = self.classifier.predict_proba(points)

        for point in points:
            assert actual[point] == expected_cls[point]

    @pytest.mark.parametrize(
        "points,expected_cls",
        [
            (np.array([np.array([2.0, 2.0, 2.0])]), np.array(["green"])),
            (np.array([np.array([3.0, 1.0, 1.0])]), np.array(["red"])),
            (np.array([np.array([0.5, 3.5, 2.5])]), np.array(["green"])),
            (
                np.array(
                    [
                        np.array([2.0, 2.0, 2.0]),
                        np.array([3.0, 1.0, 1.0]),
                        np.array([0.5, 3.5, 2.5]),
                    ]
                ),
                np.array(["green", "red", "green"]),
            ),
        ],
    )
    def test_predict(self, points: NDArray[T], expected_cls: NDArray[C]):
        assert np.all(self.classifier.predict(points)) == np.all(expected_cls)
