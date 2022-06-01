from __future__ import annotations
from typing import List, Dict, Set

from Complexes.FilteredSimplicialComplex import FilteredSimplicialComplex
from Complexes.Simplex import Simplex
from PersistentHomology.PersHomBase import PersHomBase, PersDiag, PersPoint
import numpy as np


class PersHom2(PersHomBase):
    __simplex_ordering: List[Simplex]
    __simplex_count: int
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
        low: Dict[int, int] = dict()
        self.__working_matrix = list()
        for i in self.__boundary_matrix:
            self.__working_matrix.append(i.copy())

        for i in range(self.__simplex_count):
            last_in_col = self.__get_last_in_col(i)
            while last_in_col in low and last_in_col is not None:
                competing_col = low[last_in_col]
                self.__add_col(i, competing_col)
                last_in_col = self.__get_last_in_col(i)
            if last_in_col is not None:
                low[last_in_col] = i

        self._pers_diag = PersDiag()
        for row, col in low.items():
            pers_point = PersPoint(
                born_index=row,
                die_index=col,
                born=self.__weights[row],
                die=self.__weights[col],
                dim=self.__simplex_ordering[row].dim
            )
            self._pers_diag.add_point(pers_point)
