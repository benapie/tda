from abc import ABC, abstractmethod
import numpy as np
import logging


class SNFBase(ABC):
    _matrix: np.matrix
    _snf_matrix: np.matrix | None
    _logger: logging.Logger

    def __init__(self, matrix: np.matrix):
        self._matrix = matrix
        self._snf_matrix = None
        self._logger = logging.getLogger(__name__)

    @abstractmethod
    def compute_snf(self):
        pass

    @property
    def snf_matrix(self) -> np.matrix:
        assert self._snf_matrix is not None, "Smith normal form matrix has not been computed."
        return self._snf_matrix.copy()

    @property
    def original_matrix(self) -> np.matrix:
        return self._matrix.copy()