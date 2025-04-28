from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from src.homeworks.homework2.KDTree import T


class Scaler(ABC):
    @abstractmethod
    def fit(self, data: NDArray[T]):
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: NDArray[T]) -> NDArray[np.float64]:
        raise NotImplementedError

    def fit_transform(self, data: NDArray[T]) -> NDArray[np.float64]:
        self.fit(data)
        return self.transform(data)


class MinMaxScaler(Scaler):
    def __init__(self):
        self.min: NDArray[T] | None = None
        self.max: NDArray[T] | None = None

    def fit(self, data: NDArray[T]):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data: NDArray[T]) -> NDArray[np.float64]:
        if self.min is None or self.max is None:
            raise ValueError("Scaler has not been fitted yet.")

        return np.divide(
            data - self.min,
            self.max - self.min,
            out=np.zeros_like(data, dtype=np.float64),
            where=((self.max - self.min) != 0),
        )


class MaxAbsScaler(Scaler):
    def __init__(self):
        self.max_abs: NDArray[T] | None = None

    def fit(self, data: NDArray[T]):
        self.max_abs = abs(np.max(data, axis=0))

    def transform(self, data: NDArray[T]) -> NDArray[np.float64]:
        if self.max_abs is None:
            raise ValueError("Scaler has not been fitted yet.")

        return np.divide(
            data,
            self.max_abs,
            out=np.zeros_like(data, dtype=np.float64),
            where=(self.max_abs != 0),
        )


class StandardScaler(Scaler):
    def __init__(self):
        self.mean: NDArray[T] | None = None
        self.std: NDArray[T] | None = None

    def fit(self, data: NDArray[T]):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data: NDArray[T]) -> NDArray[np.float64]:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        return np.divide(
            data - self.mean,
            self.std,
            out=np.zeros_like(data, dtype=np.float64),
            where=(self.std != 0),
        )
