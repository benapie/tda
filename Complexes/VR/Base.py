from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Callable

from Complexes.FilteredSimplicialComplex import FilteredSimplicialComplex
from Complexes.SimplicialComplex import *
from numpy import ndarray


class VRBase(ABC):
    _points: List[ndarray]
    _complex: FilteredSimplicialComplex | None
    _metric: Callable[[ndarray[float, ...], ndarray[float, ...]], float]
    _epsilon: float
    _is_skeleton_constructed: bool

    def __init__(self, points: List[ndarray], epsilon: float,
                 metric: Callable[[ndarray[float, ...], ndarray[float, ...]], float]):
        self._points = points
        self._complex = None
        self._metric = metric
        self._epsilon = epsilon
        self._is_skeleton_constructed = False

    def get_complex(self) -> FilteredSimplicialComplex:
        return self._complex

    @abstractmethod
    def compute_skeleton(self):
        pass

    @abstractmethod
    def compute_expansion(self, dim: int):
        pass
