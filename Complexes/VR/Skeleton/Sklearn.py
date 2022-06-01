from __future__ import annotations
from abc import ABC

from numpy import array

from Complexes.FilteredSimplicialComplex import FilteredSimplicialComplex
from Complexes.Simplex import Simplex
from Complexes.VR.Base import VRBase
from sklearn import neighbors


class Sklearn(VRBase, ABC):
    def compute_skeleton(self):
        self._complex = FilteredSimplicialComplex()
        for i, _ in enumerate(self._points):
            self._complex.add_simplex(Simplex({i}), 0)

        points = array(self._points)
        neighbours = neighbors.NearestNeighbors(radius=self._epsilon).fit(points)
        distances_list, indices_list = neighbours.radius_neighbors(points)
        for i, (distances, indices) in enumerate(zip(distances_list, indices_list)):
            for distance, j in zip(distances[1:], indices[1:]):
                self._complex.add_simplex(Simplex({i, j}), distance)
        self._is_skeleton_constructed = True
