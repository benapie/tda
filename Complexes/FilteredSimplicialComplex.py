from __future__ import annotations
from typing import Set, Dict, Generator, List

from Complexes.Simplex import Simplex


class FilteredSimplicialComplex:
    __simplices: Dict[int, Set[Simplex]]
    __weights: Dict[Simplex, float]
    __check_valid: bool

    def __init__(self, check_valid=False):
        self.__simplices = dict()
        self.__weights = dict()
        self.__check_valid = check_valid
        pass

    def __repr__(self) -> str:
        output = []
        for p in self.__simplices:
            for p_simplex in self.__simplices[p]:
                output.append(f"{str(p_simplex)} w: {self.__weights[p_simplex]}")
        return "\n".join(output)

    def __contains__(self, item: Simplex) -> bool:
        if not isinstance(item, Simplex):
            raise ValueError(
                "Simplicial complex only contains simplex, contains operation of non-simplex type prohibited.")
        if item.dim not in self.__simplices:
            return False
        return item in self.__simplices[item.dim]

    def __eq__(self, other: FilteredSimplicialComplex):
        if self.__simplices != other.__simplices:
            return False
        if self.__weights != other.__weights:
            return False
        return True

    @property
    def dim(self):
        return max(self.__simplices.keys())

    @property
    def size(self):
        size = 0
        for simplices in self.__simplices.values():
            size += len(simplices)
        return size

    def p_simplices(self, p: int) -> Generator[Simplex, None, None]:
        if p not in self.__simplices:
            return
        for p_simplex in self.__simplices[p]:
            yield p_simplex

    def __add_simplex(self, simplex: Simplex, weight: float):
        if simplex.dim not in self.__simplices:
            self.__simplices[simplex.dim] = set()
        self.__simplices[simplex.dim].add(simplex)
        self.__weights[simplex] = weight

    def add_simplex(self, simplex: Simplex, weight: float):
        if simplex.dim == 0:
            self.__add_simplex(simplex, weight)
            return
        if self.__check_valid:
            for facet in simplex.facets:
                if simplex.dim - 1 not in self.__simplices:
                    raise ValueError("Not all facets are in the complex, cannot add simplex.")
                if facet not in self.__simplices[simplex.dim - 1]:
                    raise ValueError("Not all facets are in the complex, cannot add simplex.")
                facet_weight = self.__weights[facet]
                if facet_weight > weight:
                    raise ValueError("Not all facets have a lower weight.")
        self.__add_simplex(simplex, weight)

    def get_weight(self, simplex: Simplex) -> float:
        if simplex.dim not in self.__simplices:
            raise ValueError("Simplex not in complex.")
        if simplex not in self.__simplices[simplex.dim]:
            raise ValueError("Simplex not in complex.")
        return self.__weights[simplex]

    def p_simplex_count(self, p: int) -> int:
        if p not in self.__simplices:
            return 0
        return len(self.__simplices[p])

    def get_edge_neighbours(self, vertex: int) -> Set[int]:
        if 1 not in self.__simplices:
            return set()
        edge_set = set()
        for edge in self.__simplices[1]:
            if vertex in edge:
                edge_set = edge_set.union(edge.get_vertices())
        edge_set.discard(vertex)
        return edge_set

    def reweight(self, simplex: Simplex, weight: float) -> None:
        if simplex.dim not in self.__simplices:
            raise ValueError("Simplex not in complex.")
        if simplex not in self.__simplices[simplex.dim]:
            raise ValueError("Simplex not in complex.")
        self.__weights[simplex] = weight

    def get_simplex_ordering(self) -> List[Simplex]:
        simplex_list = list()
        for simplices in self.__simplices.values():
            simplex_list += simplices
        simplex_list = sorted(simplex_list, key=lambda k: self.__weights[k])
        return simplex_list

    def get_weight_ordering(self) -> List[float]:
        return sorted(self.__weights.values())

    def cap(self, weight_limit: float) -> FilteredSimplicialComplex:
        fc = FilteredSimplicialComplex()
        for simplex in self.get_simplex_ordering():
            if self.__weights[simplex] > weight_limit:
                break
            fc.add_simplex(simplex, self.__weights[simplex])
        return fc
