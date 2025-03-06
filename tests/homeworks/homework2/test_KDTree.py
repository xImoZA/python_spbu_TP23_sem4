import math
import random

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from src.homeworks.homework2.KDTree import KDTree, Point, T


def brute_force_search(
    train: list[Point[T]], test: list[Point[T]], k: int
) -> dict[Point[T], list[Point[T]]]:
    knn = {}
    for test_point in test:
        neighbors = []
        for point in train:
            dist = KDTree.distance(test_point, point)
            neighbors.append((point, dist))

        neighbors = sorted(neighbors[:k], key=lambda x: x[1])
        knn[test_point] = [p[0] for p in neighbors]

    return knn


class TestKDTree:
    @given(
        st.integers(min_value=100, max_value=200),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=30),
    )
    def test_query(self, train_size, k, leaf_size, neighbours):
        x_train = [
            Point(np.array([random.randint(-100, 100) for _ in range(k)]))
            for _ in range(train_size)
        ]
        x_test = [
            Point(np.array([random.randint(-100, 100) for _ in range(k)]))
            for _ in range(30)
        ]
        kdtree = KDTree(x_train, leaf_size)

        tree_search = kdtree.query(x_test, neighbours)
        stupid_search = brute_force_search(x_train, x_test, neighbours)

        for point in x_test:
            stupid_dist = sorted(
                [KDTree.distance(point, near_point) for near_point in stupid_search]
            )
            tree_dist = sorted(
                [KDTree.distance(point, near_point) for near_point in tree_search]
            )

            for i in range(neighbours):
                assert math.isclose(stupid_dist[i], tree_dist[i])
