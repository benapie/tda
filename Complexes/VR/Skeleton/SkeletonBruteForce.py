from __future__ import annotations
from abc import ABC
from itertools import combinations
from typing import List, Callable

from numpy import ndarray

from Complexes.FilteredSimplicialComplex import FilteredSimplicialComplex
from Complexes.Simplex import Simplex
from Complexes.VR.Base import VRBase


class SkeletonBruteForce(VRBase, ABC):
    def __init__(self, points: List[ndarray], epsilon: float,
                 metric: Callable[[ndarray[float, ...], ndarray[float, ...]], float]):
        super().__init__(points, epsilon, metric)

    def compute_skeleton(self):
        self._complex = FilteredSimplicialComplex()
        for i, _ in enumerate(self._points):
            self._complex.add_simplex(Simplex({i}), 0)
        for (i, x), (j, y) in combinations(enumerate(self._points), 2):
            dist = self._metric(x, y)
            if dist < self._epsilon:
                self._complex.add_simplex(Simplex({i, j}), dist)
        self._is_skeleton_constructed = True
