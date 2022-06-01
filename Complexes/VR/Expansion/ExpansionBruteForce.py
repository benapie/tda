from __future__ import annotations
from abc import ABC
from typing import List, Callable

from numpy import ndarray
from itertools import combinations

from Complexes.VR.Base import VRBase


class ExpansionBruteForce(VRBase, ABC):
    def __init__(self, points: List[ndarray], epsilon: float,
                 metric: Callable[[ndarray[float, ...], ndarray[float, ...]], float]):
        super().__init__(points, epsilon, metric)

    def compute_expansion(self, dim: int):
        if not self._is_skeleton_constructed:
            raise ReferenceError("Skeleton not constructed.")
        for simplex_dim in range(2, dim + 1):
            for elem in combinations(self._complex.p_simplices(simplex_dim - 1), simplex_dim + 1):
                flag = True
                max_dist = 0
                for (x, y) in combinations(elem, 2):
                    if x.intersect(y).dim == -1:
                        flag = False
                        break
                    max_dist = max([max_dist, self._complex.get_weight(x), self._complex.get_weight(y)])
                if flag:
                    union = elem[0].union(*elem[1:])
                    if union.dim == simplex_dim:
                        self._complex.add_simplex(union, max_dist)
            if self._complex.p_simplex_count(simplex_dim) < simplex_dim + 1:
                break
