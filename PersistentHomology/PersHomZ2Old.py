from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Any, Generator

from Complexes.FilteredSimplicialComplex import FilteredSimplicialComplex
from Complexes.Simplex import Simplex
import numpy as np


@dataclass
class IPersPoint:
    born: int
    die: int


@dataclass
class PersPoint:
    born: float
    die: float


class PersDiagsBase(ABC):
    _data: Dict[int, List[IPersPoint | PersPoint]]

    def __init__(self):
        self._data = dict()

    @abstractmethod
    def add_point(self, point: Any, dim: int):
        pass

    @property
    def points(self) -> Generator[Tuple[IPersPoint | PersPoint, int], None, None]:
        for dim, points in self._data.items():
            for point in points:
                yield point, dim

    def diag(self, dim: int) -> List[IPersPoint | PersPoint]:
        if dim not in self._data:
            return list()
        return self._data[dim]


class IPersDiags(PersDiagsBase):
    def __init__(self):
        super().__init__()

    def add_point(self, ipers_point: IPersPoint, dim: int) -> None:
        if dim not in self._data:
            self._data[dim] = list()
        self._data[dim].append(ipers_point)

    def __repr__(self) -> str:
        output = []
        for dim in self._data:
            for point in self._data[dim]:
                output.append(f"{dim}-{point}")
        output = ",".join(output)
        output2 = ["IPersDiag(", output, ")"]
        return "".join(output2)


class PersDiags(PersDiagsBase):
    def __init__(self):
        super().__init__()

    def add_point(self, pers_point: PersPoint, dim: int) -> None:
        if dim not in self._data:
            self._data[dim] = list()
        self._data[dim].append(pers_point)

    def __repr__(self) -> str:
        output = []
        for dim in self._data:
            for point in self._data[dim]:
                output.append(f"{dim}-{point}")
        output = ",".join(output)
        output2 = ["PersDiag(", output, ")"]
        return "".join(output2)


class PersHomZ2Old:
    __simplex_ordering: List[Simplex]
    __weights: List[float]
    __boundary: np.ndarray
    __reduced_boundary: np.ndarray | None
    __simplex_count: int
    __lowest_rows: Dict[int, int]
    __ipers_diag: IPersDiags
    __pers_diag: PersDiags

    def __init__(self, filtered_complex: FilteredSimplicialComplex):
        self.__simplex_ordering = filtered_complex.get_simplex_ordering()
        self.__weights = filtered_complex.get_weight_ordering()
        self.__boundary = np.zeros((len(self.__simplex_ordering), len(self.__simplex_ordering)))
        self.__simplex_count = len(self.__simplex_ordering)
        self.__reduced_boundary = None
        self.__lowest_rows = None
        self.__ipers_diag = None
        self.__pers_diag = None
        for i, simplex in enumerate(self.__simplex_ordering):
            for j in range(i):
                if self.__simplex_ordering[j] in simplex.facets:
                    self.__boundary[j][i] = 1

    @staticmethod
    def __get_last_1_in_array(array: np.ndarray) -> int | None:
        if np.all(array == 0):
            return None
        return max(j for j, val in enumerate(array) if val == 1)

    @staticmethod
    def __add_col_op_mod_2(array: np.ndarray, col_a: int, col_b: int) -> None:
        """
        Inline operations, does col_a = col_a + col_b
        """
        array[:, col_a] = np.mod(array[:, col_a] + array[:, col_b], 2)

    def compute(self) -> None:
        self.reduce_boundary()
        self.construct_ipers_diag()
        self.construct_pers_diag()

    def reduce_boundary(self) -> None:
        low: Dict[int, int] = dict()
        boundary = self.__boundary.copy()

        for i in range(self.__simplex_count):
            last_1 = self.__get_last_1_in_array(boundary[:, i])
            if last_1 is None:
                continue
            if last_1 not in low:
                low[last_1] = i
                continue
            while last_1 in low and last_1 is not None:
                competing_col = low[last_1]
                self.__add_col_op_mod_2(boundary, i, competing_col)
                last_1 = self.__get_last_1_in_array(boundary[:, i])
            if last_1 is not None:
                low[last_1] = i

        self.__lowest_rows = low
        self.__reduced_boundary = boundary

    def construct_ipers_diag(self) -> None:
        assert self.__reduced_boundary is not None, "Reduced boundary does not seem to be computed."
        ipers_diag = IPersDiags()
        for row, col in self.__lowest_rows.items():
            ipers_diag.add_point(IPersPoint(row, col), self.__simplex_ordering[col].dim)
        self.__ipers_diag = ipers_diag

    def construct_pers_diag(self) -> None:
        assert self.__ipers_diag is not None, "Index persistent diagram is not yet constructed."
        pers_diag = PersDiags()
        for ipers_point, dim in self.__ipers_diag.points:
            pers_point = PersPoint(self.__weights[ipers_point.born], self.__weights[ipers_point.die])
            pers_diag.add_point(pers_point, dim)
        self.__pers_diag = pers_diag

    def get_boundary(self) -> np.ndarray:
        assert self.__reduced_boundary is not None, "Reduced boundary does not seem to be computed."
        return self.__reduced_boundary

    def get_ipers_diags(self) -> IPersDiags:
        assert self.__ipers_diag is not None, "Index persistence diagram does not seem to be constructed."
        return self.__ipers_diag

    def get_pers_diags(self) -> PersDiags:
        assert self.__pers_diag is not None, "Persistence diagram does not seem to be constructed."
        return self.__pers_diag

    def get_ipers_diag(self, dim: int) -> List[IPersPoint]:
        return self.__ipers_diag.diag(dim)

    def get_pers_diag(self, dim: int) -> List[PersPoint]:
        return self.__pers_diag.diag(dim)
