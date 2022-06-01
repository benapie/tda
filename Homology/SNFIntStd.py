from abc import ABC

import numpy as np

from Homology.SNFBase import SNFBase
import smithnormalform as snf
from smithnormalform import matrix, z, snfproblem


class SNFIntStd(SNFBase, ABC):
    """
    This class implements the standard algorithm for Smith normal form with integer entries.
    """
    __snf_input_matrix: snf.matrix.Matrix

    def __init__(self, input_matrix: np.matrix[int, ...]):
        super().__init__(input_matrix)
        self.__snf_input_matrix = self.__np_matrix_to_snf_input_matrix(input_matrix)
        self._logger.info("Matrix initialised into correct data type.")

    @staticmethod
    def __np_matrix_to_snf_input_matrix(input_matrix: np.matrix[int, ...]) -> snf.matrix.Matrix:
        entry_list = [snf.z.Z(int(entry)) for entry in input_matrix.A1]
        row_count = input_matrix.shape[0]
        col_count = input_matrix.shape[1]
        return snf.matrix.Matrix(row_count, col_count, entry_list)

    @staticmethod
    def __snf_input_matrix_to_np_matrix(input_matrix: snf.matrix.Matrix) -> np.matrix[int, ...]:
        entry_list = [entry.a for entry in input_matrix.elements]
        np_matrix = np.matrix(entry_list)
        np_matrix = np_matrix.reshape((input_matrix.h, input_matrix.w))
        return np_matrix

    def compute_snf(self):
        problem = snf.snfproblem.SNFProblem(self.__snf_input_matrix)
        problem.computeSNF()
        assert problem.isValid(), "Problem is not valid (library error)."
        self._logger.info("SNF computed.")
        self._snf_matrix = self.__snf_input_matrix_to_np_matrix(problem.J)
        self._logger.info("SNF matrix casted to np.matrix data type.")

