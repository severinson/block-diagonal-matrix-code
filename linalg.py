############################################################################
# Copyright 2017 Albin Severinson                                          #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

""" This module contains the needed linear algebra functionality that is not
provided by Numpy etc. These functions are used mainly for code construction. """

import unittest
import numpy as np
import scipy as sp

class LinalgError(Exception):
    """ Base class for exceptions in this module. """
    def __init__(self, message):
        super().__init__(message)

def null(matrix, eps=1e-12):
    """ Return a matrix whose columns span the null space of matrix.

    Source:
    https://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix

    Args:
    matrix: The matrix to compute the null space of.

    Returns:
    A matrix whose columns span the null space of the input matrix.
    """
    _, s, vh = np.linalg.svd(matrix)
    padding = max(0, np.shape(matrix)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,), dtype=bool)), axis=0)
    null_space = sp.compress(null_mask, vh, axis=0)
    return sp.transpose(null_space)

def construct_bdm(rows, cols, blocks):
    """ Construct a block-diagonal encoding matrix with rows rows and cols
    columns from blocks blocks.

    Each block is a real-numbered matrix whose elements are generated randomly
    from a Gaussian distribution with zero mean and unit variance.

    Args:
    rows: Number of rows of the resulting encoding matrix.
    cols: Number of columns of the resulting encoding matrix.
    blocks: Number of blocks.

    Returns:
    The resulting encoding matrix.

    Raises:
    LinalgError: If the paramaters are invalid.
    """

    if rows % blocks != 0:
        raise LinalgError('rows must be divisible by blocks.')
    if cols % blocks != 0:
        raise LinalgError('cols must be divisible by blocks.')

    block_rows = int(rows / blocks)
    block_cols = int(cols / blocks)
    block = np.random.randn(block_rows, block_cols)
    bdm = np.zeros([rows, cols])
    for i in range(blocks):
        bdm[i*block_rows:(i+1)*block_rows, i*block_cols:(i+1)*block_cols] = block

    return bdm, block


class Tests(unittest.TestCase):
    """ Module unit tests. """

    def test_null(self):
        """ Test that the code can find the null space of a matrix. """
        eps = 1e-12
        matrix = np.array([[2, 3, 4], [-4, 2, 3]])
        self.assertTrue(np.dot(matrix, null(matrix)).sum() < eps)

        matrix = matrix.T
        self.assertTrue(np.dot(matrix, null(matrix)).sum() < eps)

    def test_bdm_null(self):
        """ Test that the code can find the null space of a BDM matrix. """
        eps = 1e-12
        bdm, _ = construct_bdm(5, 4, 1)
        bdm_null = null(bdm.T)
        self.assertTrue(np.dot(bdm_null.T, bdm).sum() < eps)

        bdm, _ = construct_bdm(12, 8, 4)
        bdm_null = null(bdm.T)
        self.assertTrue(np.dot(bdm_null.T, bdm).sum() < eps)

        with self.assertRaises(LinalgError):
            construct_bdm(12, 8, 8)

if __name__ == '__main__':
    unittest.main()
