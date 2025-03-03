import heapq
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import numpy as np

T = TypeVar("T", bound=float)


@dataclass
class Node(Generic[T]):
    axis: Optional[int] = None
    median: Optional[T] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    points: Optional[list[tuple[T, ...]]] = None
    is_leaf: bool = field(init=False)

    def __post_init__(self):
        self.is_leaf = self.points is not None


class KDTree:
    def __init__(self, points: list[tuple[T, ...]], leaf_size: int = 1):
        if leaf_size <= 0:
            raise ValueError("leaf_size must by greater then 0")
        if len(points) == 0:
            raise ValueError("Must be at least one point")

        self.leaf_size: int = leaf_size
        self.root: Node[T] = self._build_kdtree(points)

    @staticmethod
    def _select_axis(points: list[tuple[T, ...]]) -> int:
        variances = np.var(points, axis=0)
        return int(np.argmax(variances))

    def _build_kdtree(self, points: list[tuple[T, ...]]) -> Node[T]:
        if len(points) <= self.leaf_size:
            return Node(points=points)

        axis = KDTree._select_axis(points)

        points.sort(key=lambda point: point[axis])
        median = len(points) // 2

        return Node(
            axis=axis,
            median=points[median][axis],
            left=self._build_kdtree(points[:median]),
            right=self._build_kdtree(points[median:]),
        )

    def query(
        self, points: list[tuple[T]], k: int
    ) -> dict[tuple[T], list[tuple[T, ...]]]:
        if len(points) == 0:
            raise ValueError("Points list must not be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        knn = {}
        for point in points:
            knn_point = KDTree._search(point, k, self.root, [])
            knn[point] = [heapq.heappop(knn_point)[1] for _ in range(k)]
        return knn

    @staticmethod
    def _search(
        target: tuple[T],
        k: int,
        node: Optional[Node[T]],
        neighbors: list[tuple[float, tuple[T, ...]]],
    ) -> list[tuple[float, tuple[T, ...]]]:
        if node:
            if node.is_leaf and node.points:
                for point in node.points:
                    if len(point) != len(target):
                        raise ValueError("Points must have the same dimensionality")

                    dist: float = KDTree.distance(target, point)

                    if len(neighbors) < k:
                        heapq.heappush(neighbors, (-dist, point))
                    elif dist < -neighbors[0][0]:
                        heapq.heappushpop(neighbors, (-dist, point))

                return neighbors

            if node.axis is None or node.median is None:
                raise ValueError(
                    "If Node is not leaf, axis and median must be not None"
                )

            axis = node.axis
            if target[axis] < node.median:
                next_node = node.left
                other_node = node.right
            else:
                next_node = node.right
                other_node = node.left

            neighbors = KDTree._search(target, k, next_node, neighbors)
            if len(neighbors) < k or abs(target[axis] - node.median) < -neighbors[0][0]:
                neighbors = KDTree._search(target, k, other_node, neighbors)

        return neighbors

    @staticmethod
    def distance(point1: tuple[T, ...], point2: tuple[T, ...]) -> float:
        if len(point1) != len(point2):
            raise ValueError("Points must have the same dimensionality")

        return np.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))
