from __future__ import annotations
from abc import ABC
from itertools import combinations
from typing import List, Callable, Set, Hashable

from numpy import ndarray

from Complexes.Simplex import Simplex
from Complexes.VR.Base import VRBase


class Inductive(VRBase, ABC):
    def __init__(self, points: List[ndarray], epsilon: float,
                 metric: Callable[[ndarray[float, ...], ndarray[float, ...]], float]):
        super().__init__(points, epsilon, metric)

    def lower_neighbours(self, v: int) -> Set[int]:
        edge_neighbours = self._complex.get_edge_neighbours(v)
        lower_edge_neighbours = set()
        for neighbour in edge_neighbours:
            if neighbour < v:
                lower_edge_neighbours.add(neighbour)
        return lower_edge_neighbours

    def compute_weights(self):
        for p in range(2, self._complex.dim + 1):
            for p_simplex in self._complex.p_simplices(p):
                weight = max(self._complex.get_weight(facet) for facet in p_simplex.facets)
                self._complex.reweight(p_simplex, weight)

    def compute_expansion(self, dim: int):
        for simplex_dim in range(2, dim + 1):
            stack = list(self._complex.p_simplices(simplex_dim - 1))
            for simplex in self._complex.p_simplices(simplex_dim - 1):
                common_lower_neighbours = None
                for vertex in simplex:
                    if common_lower_neighbours is None:
                        common_lower_neighbours = self.lower_neighbours(vertex)
                    else:
                        common_lower_neighbours = common_lower_neighbours.intersection(self.lower_neighbours(vertex))
                for neighbour in common_lower_neighbours:
                    self._complex.add_simplex(simplex + neighbour, 0)
        self.compute_weights()
