import heapq
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import numpy as np

T = TypeVar("T", bound=float)


@dataclass
class Point(Generic[T]):
    coord: tuple[T, ...]
    cls: Optional[int]

    def __lt__(self, other):
        if isinstance(other, Point):
            return self.coord < other.coord
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.coord == other.coord
        return NotImplemented

    def __hash__(self):
        return hash(self.coord)


@dataclass
class Node(Generic[T]):
    axis: Optional[int] = None
    median: Optional[T] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    points: list[Point[T]] | None = None
    is_leaf: bool = field(init=False)

    def __post_init__(self):
        self.is_leaf = self.points is not None


class KDTree:
    def __init__(self, points: list[Point[T]], leaf_size: int = 1):
        if leaf_size <= 0:
            raise ValueError("leaf_size must by greater then 0")
        if len(points) == 0:
            raise ValueError("Must be at least one point")

        self.leaf_size: int = leaf_size
        self.root: Node[T] = self._build_kdtree(points)

    @staticmethod
    def _select_axis(points: list[Point[T]]) -> int:
        variances = np.var([p.coord for p in points], axis=0)
        return int(np.argmax(variances))

    def _build_kdtree(self, train: list[Point[T]]) -> Node[T]:
        if len(train) <= self.leaf_size:
            return Node(points=train)

        axis = KDTree._select_axis(train)

        train.sort(key=lambda point: point.coord[axis])
        median = len(train) // 2

        return Node(
            axis=axis,
            median=train[median].coord[axis],
            left=self._build_kdtree(train[:median]),
            right=self._build_kdtree(train[median:]),
        )

    def query(self, points: list[Point[T]], k: int) -> dict[Point[T], list[Point[T]]]:
        if len(points) == 0:
            raise ValueError("Points list must not be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        knn: dict[Point[T], list[Point[T]]] = {}
        for point in points:
            knn_point: list[tuple[float, Point[T]]] = KDTree._search(
                point, k, self.root, []
            )
            knn[point] = [heapq.heappop(knn_point)[1] for _ in range(k)]
        return knn

    @staticmethod
    def _search(
        target: Point[T],
        k: int,
        node: Node[T] | None,
        neighbors: list[tuple[float, Point[T]]],
    ) -> list[tuple[float, Point[T]]]:
        if node:
            if node.is_leaf and node.points:
                for point in node.points:
                    if len(point.coord) != len(target.coord):
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
            if target.coord[axis] < node.median:
                next_node = node.left
                other_node = node.right
            else:
                next_node = node.right
                other_node = node.left

            neighbors = KDTree._search(target, k, next_node, neighbors)
            if (
                len(neighbors) < k
                or abs(target.coord[axis] - node.median) < -neighbors[0][0]
            ):
                neighbors = KDTree._search(target, k, other_node, neighbors)

        return neighbors

    @staticmethod
    def distance(point1: Point[T], point2: Point[T]) -> float:
        if len(point1.coord) != len(point2.coord):
            raise ValueError("Points must have the same dimensionality")

        return np.sqrt(sum((x - y) ** 2 for x, y in zip(point1.coord, point2.coord)))
