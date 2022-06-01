from __future__ import annotations
from typing import List, Dict

from Complexes.FilteredSimplicialComplex import FilteredSimplicialComplex
from Complexes.Simplex import Simplex
from PersistentHomology.PersHomBase import PersHomBase, PersDiag, PersPoint
import numpy as np


class PersHom1(PersHomBase):
    __simplex_ordering: List[Simplex]
    __simplex_count: int
    __weights: List[float]
    __boundary_matrix: np.ndarray
    __working_matrix: np.ndarray | None

    def __init__(self, filtered_complex: FilteredSimplicialComplex):
        super().__init__()
        self.__simplex_ordering = filtered_complex.get_simplex_ordering()
        self.__simplex_to_order = dict()
        for i, simplex in enumerate(self.__simplex_ordering):
            self.__simplex_to_order[simplex] = i
        self.__simplex_count = len(self.__simplex_ordering)
        self.__weights = filtered_complex.get_weight_ordering()
        self.__boundary_matrix = np.zeros((len(self.__simplex_ordering), len(self.__simplex_ordering)))
        for i, simplex in enumerate(self.__simplex_ordering):
            for facet in simplex.facets:
                if facet not in self.__simplex_to_order:
                    if facet.dim != -1:
                        raise Exception()
                    continue
                j = self.__simplex_to_order[facet]
                self.__boundary_matrix[j][i] = 1
        self.__working_matrix = None

    def __get_last_in_col(self, i: int) -> int | None:
        if np.all(self.__working_matrix[:, i] == 0):
            return None
        return max(j for j, val in enumerate(self.__working_matrix[:, i]) if val == 1)

    def __add_col(self, i: int, j: int) -> None:
        """
        R_i <- R_i + R_j
        """
        assert i < self.__simplex_count and j < self.__simplex_count, "Column index out of range."
        assert self.__working_matrix is not None, "Working matrix is not initialised."
        self.__working_matrix[:, i] = np.mod(self.__working_matrix[:, i] + self.__working_matrix[:, j], 2)

    def compute(self) -> None:
        low: Dict[int, int] = dict()
        self.__working_matrix = self.__boundary_matrix.copy()

        for i in range(self.__simplex_count):
            last_in_col = self.__get_last_in_col(i)
            if last_in_col is None:
                continue
            if last_in_col not in low:
                low[last_in_col] = i
                continue
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
