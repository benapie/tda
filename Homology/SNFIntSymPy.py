from abc import ABC

import numpy as np
import sympy as sp
from sympy.matrices.normalforms import smith_normal_form
from Homology.SNFBase import SNFBase


class SNFIntSymPy(SNFBase, ABC):
    """
    This class implements the standard algorithm for Smith normal form with integer entries.
    """
    __snf_input_matrix: sp.Matrix

    def __init__(self, input_matrix: np.matrix[int, ...]):
        super().__init__(input_matrix)
        self.__snf_input_matrix = self.__np_matrix_to_snf_input_matrix(input_matrix)
        self._logger.info("Matrix initialised into correct data type.")

    @staticmethod
    def __np_matrix_to_snf_input_matrix(input_matrix: np.matrix[int, ...]) -> sp.Matrix:
        entry_list = input_matrix.tolist()
        return sp.Matrix(entry_list)

    @staticmethod
    def __snf_input_matrix_to_np_matrix(input_matrix: sp.Matrix) -> np.matrix[int, ...]:
        np_matrix = np.matrix(input_matrix)
        return np_matrix

    def compute_snf(self):
        snf = sp.matrices.normalforms.smith_normal_form(self.__snf_input_matrix)
        self._logger.info("SNF computed.")
        print(snf)
        self._logger.info("SNF matrix casted to np.matrix data type.")

