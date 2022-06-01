from __future__ import annotations
from random import randint
from typing import List, Callable, Tuple
from numpy import ndarray, sqrt, inf, array
from itertools import combinations
import miniball


class MiniBall2D:
    __points: List[ndarray]
    __dim: int

    def __init__(self, points: List[ndarray], d: Callable[[ndarray[float, float], ndarray[float, float]], float]):
        for point in points:
            if point.size != 2:
                raise ValueError("Points must have size 2.")
        self.__points = points
        self.__metric = d

    def compute(self):
        C, r2 = miniball.get_bounding_ball(array(self.__points), epsilon=0)
        r = sqrt(r2)
        for point in self.__points:
            print(r - self.__metric(C, point))
        return C, r2

