from __future__ import annotations
from typing import List, Dict, Set

from Complexes.FilteredSimplicialComplex import FilteredSimplicialComplex
from Complexes.Simplex import Simplex
from PersistentHomology.PersHomBase import PersHomBase, PersDiag, PersPoint
import numpy as np
import bisect


class PersHom4(PersHomBase):
    __simplex_ordering: List[Simplex]
    __simplex_to_order: Dict[Simplex, int]
    __simplex_count: int
    __dim: int
    __weights: List[float]
    __boundary_matrix: List[Set[int]]
    __working_matrix: List[Set[int]] | None

    def __init__(self, filtered_complex: FilteredSimplicialComplex):
        super().__init__()
        self.__simplex_ordering = filtered_complex.get_simplex_ordering()
        self.__simplex_to_order = dict()
        for i, simplex in enumerate(self.__simplex_ordering):
            self.__simplex_to_order[simplex] = i
        self.__simplex_count = len(self.__simplex_ordering)
        self.__dim = filtered_complex.dim
        self.__weights = filtered_complex.get_weight_ordering()
        self.__boundary_matrix = []
        for i, simplex in enumerate(self.__simplex_ordering):
            self.__boundary_matrix.append(set())
            for facet in simplex.facets:
                if facet not in self.__simplex_to_order:
                    if facet.dim != -1:
                        raise Exception()
                    continue
                j = self.__simplex_to_order[facet]
                self.__boundary_matrix[i].add(j)
        self.__coboundary_matrix = list()
        for i in range(self.__simplex_count):
            self.__coboundary_matrix.append(set())
        for col_i, row_is in enumerate(self.__boundary_matrix):
            for row_i in row_is:
                self.__coboundary_matrix[self.__simplex_count - 1 - row_i] \
                    .add(self.__simplex_count - 1 - col_i)
        self.__working_matrix = None

    def __get_last_in_col(self, i: int) -> int | None:
        if not self.__working_matrix[i]:
            return None
        return max(j for j in self.__working_matrix[i])

    def __add_col(self, i: int, j: int) -> None:
        """
        R_i <- R_i + R_j
        """
        for k in self.__working_matrix[j]:
            if k not in self.__working_matrix[i]:
                self.__working_matrix[i].add(k)
                continue
            self.__working_matrix[i].remove(k)

    def compute(self) -> None:
        low: Dict[int, List[int]] = dict()
        self.__working_matrix = list()
        cocycles = []
        marked = set()
        for i in range(self.__simplex_count):
            cocycles.append({i})

        self._pers_diag = PersDiag()

        for i in range(self.__simplex_count - 1, -1, -1):
            indices = set()
            for j in self.__boundary_matrix[self.__simplex_count - 1 - i]:
                j_index = self.__simplex_count - 1 - j
                if j_index in marked:
                    continue
                for k in cocycles[j_index]:
                    if k in indices:
                        indices.remove(k)
                        continue
                    indices.add(k)
            if not indices:
                continue
            marked.add(i)
            p = min(indices)
            marked.add(p)

            # print(cocycles)
            row_i = self.__simplex_count - 1 - p
            col_i = self.__simplex_count - 1 - i
            pers_point = PersPoint(
                born_index=row_i,
                die_index=col_i,
                born=self.__weights[row_i],
                die=self.__weights[col_i],
                dim=self.__simplex_ordering[row_i].dim
            )
            self._pers_diag.add_point(pers_point)

            for j in indices:
                if j == p:
                    continue
                cocycles[j].add(p)
