from __future__ import annotations
from typing import Dict
from Complexes.Simplex import *


class SimplicialComplex:
    __simplices: Dict[int, Set[Simplex]]  # Key represents simplex dimension.
    __check_valid: bool

    def __init__(self, check_valid=False):
        self.__simplices = dict()
        self.__check_valid = check_valid
        pass

    def __repr__(self):
        output = []
        for p in self.__simplices:
            for p_simplex in self.__simplices[p]:
                output.append(str(p_simplex))
        return "\n".join(output)

    def __contains__(self, item: Simplex) -> bool:
        if not isinstance(item, Simplex):
            raise ValueError(
                "Simplicial complex only contains simplex, contains operation of non-simplex type prohibited.")
        if item.dim not in self.__simplices:
            return False
        return item in self.__simplices[item.dim]

    @property
    def dim(self):
        return max(self.__simplices.keys())

    @property
    def size(self):
        size = 0
        for simplices in self.__simplices.values():
            size += len(simplices)
        return size

    def __add_vertex(self, simplex: Simplex):
        if simplex.dim not in self.__simplices:
            self.__simplices[simplex.dim] = set()
        self.__simplices[simplex.dim].add(simplex)

    def add_simplex(self, simplex: Simplex):
        if simplex.dim == 0:
            self.__add_vertex(simplex)
            return
        if self.__check_valid:
            for facet in simplex.facets:
                if simplex.dim - 1 not in self.__simplices:
                    raise ValueError("Not all facets are in the complex, cannot add simplex.")
                if facet not in self.__simplices[simplex.dim - 1]:
                    raise ValueError("Not all facets are in the complex, cannot add simplex.")
        self.__add_vertex(simplex)
