from __future__ import annotations
from abc import ABC
from itertools import combinations
from typing import List, Callable, Set, Hashable

from numpy import ndarray

from Complexes.FilteredSimplicialComplex import FilteredSimplicialComplex
from Complexes.Simplex import Simplex
from Complexes.VR.Base import VRBase


class Incremental(VRBase, ABC):
    def __init__(self, points: List[ndarray], epsilon: float,
                 metric: Callable[[ndarray[float, ...], ndarray[float, ...]], float]):
        super().__init__(points, epsilon, metric)
        self.__new_complex = FilteredSimplicialComplex()

    def lower_neighbours(self, v: int) -> Set[int]:
        edge_neighbours = self._complex.get_edge_neighbours(v)
        lower_edge_neighbours = set()
        for neighbour in edge_neighbours:
            if neighbour < v:
                lower_edge_neighbours.add(neighbour)
        return lower_edge_neighbours

    def add_cofaces(self, level: int, simplex: Simplex, lower_neighbours: Set[int]):
        stack = [(simplex, lower_neighbours)]
        while stack:
            current_simplex, current_neighbours = stack.pop()
            self.__new_complex.add_simplex(current_simplex, 0)

            if current_simplex.dim < level:
                for neighbour in current_neighbours:
                    coface = current_simplex + neighbour
                    new_lower_neighbours = current_neighbours.intersection(self.lower_neighbours(neighbour))
                    stack.append((coface, new_lower_neighbours))

    def compute_weights(self):
        for edge in self.__new_complex.p_simplices(1):
            (vertex_x, vertex_y) = edge.get_vertices()
            weight = self._metric(self._points[vertex_x], self._points[vertex_y])
            self.__new_complex.reweight(edge, weight)

        for p in range(2, self.__new_complex.dim + 1):
            for p_simplex in self.__new_complex.p_simplices(p):
                weight = max(self.__new_complex.get_weight(facet) for facet in p_simplex.facets)
                self.__new_complex.reweight(p_simplex, weight)

    def compute_expansion(self, dim: int):
        if self._complex.p_simplex_count(0) == 0:
            return
        for simplex in self._complex.p_simplices(0):
            (vertex,) = simplex.get_vertices()
            lower_neighbours = self.lower_neighbours(vertex)
            self.add_cofaces(dim, simplex, lower_neighbours)
        self.compute_weights()
        self._complex = self.__new_complex
