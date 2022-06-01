from unittest import TestCase

from Complexes.FilteredSimplicialComplex import FilteredSimplicialComplex
from Complexes.Simplex import Simplex
from PersistentHomology.PersHom1 import PersHom1
from PersistentHomology.PersHom2 import PersHom2
import itertools
import time

from PersistentHomology.PersHom3 import PersHom3
from PersistentHomology.PersHom4 import PersHom4


class SpeedTest1v2(TestCase):
    def test_12(self):
        fc = FilteredSimplicialComplex(check_valid=True)
        fc.add_simplex(Simplex(set()), -1)
        dim = 10
        vertices = list(range(dim))
        for l in range(1, dim + 1):
            for elem in itertools.combinations(vertices, l):
                fc.add_simplex(Simplex(set(elem)), l)

        pers_hom_1 = PersHom1(fc)
        pers_hom_2 = PersHom2(fc)

        start = time.perf_counter()
        pers_hom_1.compute()
        end = time.perf_counter()
        print(f"PersHom1: {round(end - start, 5)}")

        start = time.perf_counter()
        pers_hom_2.compute()
        end = time.perf_counter()
        print(f"PersHom2: {round(end - start, 5)}")

    def test_23(self):
        fc = FilteredSimplicialComplex(check_valid=True)
        fc.add_simplex(Simplex(set()), -1)
        dim = 10
        vertices = list(range(dim))
        for l in range(1, dim + 1):
            for elem in itertools.combinations(vertices, l):
                fc.add_simplex(Simplex(set(elem)), l)

        pers_hom_2 = PersHom2(fc)
        pers_hom_3 = PersHom3(fc)

        start = time.perf_counter()
        pers_hom_2.compute()
        end = time.perf_counter()
        print(f"PersHom2: {round(end - start, 5)}")

        start = time.perf_counter()
        pers_hom_3.compute()
        end = time.perf_counter()
        print(f"PersHom3: {round(end - start, 5)}")


class Test(TestCase):
    def test(self):
        fc = FilteredSimplicialComplex(check_valid=True)
        # fc.add_simplex(Simplex(set()), -1)
        # fc.add_simplex(Simplex({1}), 1)
        # fc.add_simplex(Simplex({2}), 2)
        # fc.add_simplex(Simplex({3}), 3)
        # fc.add_simplex(Simplex({1, 2}), 4)
        # fc.add_simplex(Simplex({1, 3}), 5)
        # fc.add_simplex(Simplex({2, 3}), 6)
        # fc.add_simplex(Simplex({1, 2, 3}), 7)
        # fc.add_simplex(Simplex({5}), 7)
        # fc.add_simplex(Simplex({3, 5}), 7)
        # fc.add_simplex(Simplex({4}), 7)
        # fc.add_simplex(Simplex({4, 5}), 7)


        fc = FilteredSimplicialComplex(check_valid=True)
        fc.add_simplex(Simplex(set()), -1)
        dim = 12
        vertices = list(range(dim))
        for l in range(1, dim + 1):
            for elem in itertools.combinations(vertices, l):
                fc.add_simplex(Simplex(set(elem)), l)

        start = time.perf_counter()
        pers_hom_3 = PersHom3(fc)
        pers_hom_3.compute()
        end = time.perf_counter()
        print(f"3: {round(end - start, 3)}")
        # for pers_point in pers_hom_3.get_pers_diag().points:
        #     print(pers_point)
        # print()

        start = time.perf_counter()
        pers_hom_4 = PersHom4(fc)
        pers_hom_4.compute()
        # for pers_point in pers_hom_4.get_pers_diag().points:
        #     print(pers_point)
        end = time.perf_counter()
        print(f"4: {round(end - start, 3)}")

        print(pers_hom_4.get_pers_diag() == pers_hom_3.get_pers_diag())


