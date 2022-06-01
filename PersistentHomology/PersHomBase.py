from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generator, Set, Counter
import collections


@dataclass(frozen=True)
class PersPoint:
    born_index: int
    die_index: int
    born: float
    die: float
    dim: int


class PersDiag:
    __points: Dict[int, Counter[PersPoint]]
    __point_count: int

    def __init__(self):
        self.__points = dict()
        self.__point_count = 0

    def __repr__(self) -> str:
        return f"PersDiag with {self.__point_count} points"

    def __eq__(self, other: PersDiag):
        if self.__point_count != other.__point_count:
            return False
        if self.__points.keys() != other.__points.keys():
            return False
        for key in self.__points.keys():
            if self.__points[key] != other.__points[key]:
                return False
        return True

    @property
    def points(self) -> Generator[PersPoint, None, None]:
        for p_pers_points in self.__points.values():
            for p_pers_point in p_pers_points:
                yield p_pers_point

    def p_points(self, p: int) -> Generator[PersPoint, None, None]:
        if p not in self.__points:
            return
        for pers_point in self.__points[p]:
            yield pers_point

    def add_point(self, point: PersPoint) -> None:
        if point.dim not in self.__points:
            self.__points[point.dim] = collections.Counter()
        self.__points[point.dim][point] += 1
        self.__point_count += 1


class PersHomBase(ABC):
    _pers_diag: PersDiag | None

    def __init__(self):
        self._pers_diag = None

    @abstractmethod
    def compute(self) -> None:
        pass

    def get_pers_diag(self) -> PersDiag:
        assert self._pers_diag is not None, "It seems like that the persistent diagram has not been constructed."
        return self._pers_diag
