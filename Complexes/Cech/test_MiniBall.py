from random import random
from typing import List
from unittest import TestCase

from numpy import ndarray, array, sqrt, square

from MiniBall2D import MiniBall2D


def random_points(n: int) -> List[ndarray[float, ...]]:
    points = list()
    for _ in range(n):
        points.append(array([random(), random()]))
    return points


def d(x: ndarray[float, float], y: ndarray[float, float]) -> float:
    total: float = 0
    for xx, yy in zip(x, y):
        total += square((xx - yy))
    return sqrt(total)


class TestMiniBall2D(TestCase):
    def setUp(self):
        self.miniball = MiniBall2D(random_points(5), d)


class TestCompute2D(TestMiniBall2D):
    def test(self):
        print(self.miniball.compute())
