from __future__ import annotations

from typing import List, Generator, Hashable, FrozenSet, Set
from itertools import combinations


class Simplex:
    __vertices: FrozenSet[int]

    def __init__(self, vertices: Set[int] | FrozenSet[int]):
        if isinstance(vertices, FrozenSet):
            self.__vertices = vertices
        else:
            self.__vertices = frozenset(vertices)

    def __repr__(self):
        output = [str(self.dim), "-Sim("]
        output2 = []
        for vertex in self.vertices:
            output2.append(str(vertex))
        output.append(",".join(output2))
        output.append(")")
        return "".join(output)

    def __hash__(self) -> int:
        return hash(self.__vertices)

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __contains__(self, item: Hashable) -> bool:
        return item in self.__vertices

    def __iter__(self) -> Generator[int, None, None]:
        for vertex in self.__vertices:
            yield vertex

    def __add__(self, other: int):
        vertices = set(self.__vertices)
        vertices.add(other)
        return Simplex(vertices)

    @property
    def dim(self):
        return len(self.__vertices) - 1

    @property
    def vertices(self):
        for vertex in self.__vertices:
            yield vertex

    @property
    def facets(self) -> Generator[Simplex, None, None]:
        for vertex in self.vertices:
            yield Simplex(self.__vertices.difference({vertex}))

    @property
    def faces(self) -> Generator[Simplex, None, None]:
        for face_dim in range(self.dim, 0, -1):
            for face in combinations(self.__vertices, face_dim):
                yield face

    def p_faces(self, p: int) -> Generator[Simplex, None, None]:
        for p_face in combinations(self.__vertices, p + 1):
            yield Simplex(p_face)

    def intersect(self, other: Simplex) -> Simplex:
        simplex = Simplex(self.__vertices.intersection(other.__vertices))
        return simplex

    def union(self, *simplices: Simplex) -> Simplex:
        vertices = self.__vertices
        for simplex in simplices:
            vertices = vertices.union(simplex.__vertices)
        return Simplex(vertices)

    def get_vertices(self) -> Set[int]:
        return set(self.__vertices)
