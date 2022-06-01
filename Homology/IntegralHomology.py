from Complexes.SimplicialComplex import SimplicialComplex


class IntegralHomology:
    __complex: SimplicialComplex

    def __init__(self, simplicial_complex: SimplicialComplex):
        self.__complex = simplicial_complex

    