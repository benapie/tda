import numpy as np
import logging

from Homology.SNFIntStd import SNFIntStd
from Homology.SNFIntSymPy import SNFIntSymPy

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    mat = np.matrix(np.random.randint(-1, 1, (50, 50)))
    # mat = np.matrix([[1, 2, 3], [3, 4, 3]])
    obj = SNFIntSymPy(mat)
    obj.compute_snf()
