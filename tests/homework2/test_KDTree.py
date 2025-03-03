import math

import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from src.homeworks.homework2.KDTree import KDTree, T


def brute_force_search(
    train: list[tuple[T]], test: list[tuple[T, ...]], k: int
) -> dict[tuple[T, ...], list[tuple[T]]]:
    knn = {}
    for test_point in test:
        neighbors = []
        for point in train:
            dist = KDTree.distance(test_point, point)
            neighbors.append((point, dist))

        neighbors = sorted(neighbors, key=lambda x: x[1])[:k]
        knn[test_point] = [p[0] for p in neighbors]

    return knn


class TestKDTree:
    @given(
        st.integers(min_value=100, max_value=200),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=30),
    )
    def test_query(self, size, k, leaf_size, near):
        x_train = [tuple(row) for row in np.random.randint(-100, 100, size=(size, k))]
        x_test = [tuple(row) for row in np.random.randint(-100, 100, size=(30, k))]

        kdtree = KDTree(x_train, leaf_size)
        tree_search = kdtree.query(x_test, near)
        stupid_search = brute_force_search(x_train, x_test, near)

        for point in x_test:
            stupid_dist = sorted(
                KDTree.distance(point, near_point) for near_point in stupid_search
            )
            tree_dist = sorted(
                KDTree.distance(point, near_point) for near_point in tree_search
            )

            for i in range(near):
                assert math.isclose(stupid_dist[i], tree_dist[i], rel_tol=1e-9)
