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

""" This module is an implementation of the erasure code presented by
Severinson et al. in the paper 'Block-Diagonal Coding for Distributed Computing
With Straggling Servers'. The code provides efficient encoding and erasure
correction by exploiting a block-diagonal encoding matrix.

The paper is available at https://arxiv.org/abs/1701.06631
"""

import unittest
from multiprocessing import Pool
import numpy as np
import scipy as sp

class CodingError(Exception):
    """ Base class for exceptions in this module. """
    pass

class EncoderError(CodingError):
    """ Exception raised for errors during encoding. """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class DecoderError(CodingError):
    """ Exception raised for errors during decoding. """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class BDMEncoder(object):
    """ Encoder/decoder class based on a block-diagonal encoding matrix. """
    def __init__(self, rows, cols, blocks):
        """ Create block-diagonal matrix encoder.

        Args:
        rows: Number of rows of the resulting encoding matrix.
        cols: Number of columns of the resulting encoding matrix.
        blocks: Number of blocks.
        """
        self.rows = rows
        self.cols = cols
        self.blocks = blocks
        self.bdm, self.block = self.construct_bdm(rows, cols, blocks)
        self.block_rows = int(rows / blocks)
        self.block_cols = int(cols / blocks)
        return

    @staticmethod
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
        EncoderError: If the paramaters are invalid.
        """

        if rows % blocks != 0:
            raise EncoderError('rows must be divisible by blocks.')
        if cols % blocks != 0:
            raise EncoderError('cols must be divisible by blocks.')

        block_rows = int(rows / blocks)
        block_cols = int(cols / blocks)
        block = np.random.randn(block_rows, block_cols)
        bdm = np.zeros([rows, cols])
        for i in range(blocks):
            bdm[i*block_rows:(i+1)*block_rows, i*block_cols:(i+1)*block_cols] = block

        return bdm, block

    def encode(self, source):
        """ Encode the source matrix

        Args:
        source: Matrix to encode

        Returns:
        The encoded matrix.

        Raises:
        EncoderError: If the input could not be encoded.
        """
        source_rows = source.shape[0]
        source_cols = source.shape[1]
        if source_rows != self.cols:
            raise EncoderError('Source matrix must have rows equal to the ' \
                               'number of columns of the encoding matrix.')
        output = np.dot(self.bdm, source)
        assert output.shape == (self.rows, source_cols), \
            'Encoding resulted in a matrix of incorrect shape.'
        return output

    def decodeable(self, erasure_mask):
        """ Return true if decoding is possible for the given erasure mask. """
        erasure_mask = np.array(erasure_mask, dtype=bool)

        if len(erasure_mask) != self.rows:
            raise DecoderError('The erasure mask must have length equal to ' \
                               'the number of encoding matrix rows.')

        block_distance = self.block_rows - self.block_cols
        for i in range(self.blocks):
            block_erasure = erasure_mask[i * self.block_rows:(i+1) * self.block_rows]
            if block_erasure.sum() > block_distance:
                return False

        return True

    def decode_block(self, output_erasure):
        """ Recover a single block.

        Args:
        output_erausre: A tuple where the first element is the submatrix of
        the encoded matrix to decode and the second is the erasure mask
        corresponding to the output block.

        Returns:
        The decoded block.
        """
        assert isinstance(output_erasure, tuple)
        output_block = output_erasure[0]
        block_erasure_mask = output_erasure[1]
        block_index = np.invert(block_erasure_mask)
        block = self.block[block_index, :]
        block = block[0:self.block_cols]
        output_block = output_block[0:self.block_cols]
        source_block = np.linalg.solve(block, output_block)
        return source_block

    def decode(self, output, erasure_mask, processes=4):
        """ Recover the input from erasures indicated by the erasure mask.

        Args:
        output: The encoded matrix to decode.
        erasure_mask: A mask indicating which rows of the encoded data that
        have been erased. Must be list-like with length equal to the number of
        encoded rows.
        workers: The number of worker processes to use.

        Returns:
        The decoded matrix.

        Raises:
        DecoderError: If decoding was not successfull a DecoderError is raised.
        """
        erasure_mask = np.array(erasure_mask, dtype=bool)
        if len(erasure_mask) != self.rows:
            raise DecoderError('The erasure mask must have length equal ' \
                               'to the number of encoding matrix rows.')

        if output.shape[0] != len(erasure_mask) - erasure_mask.sum():
            raise DecoderError('The output matrix must have rows equals to the ' \
                               'number of non-zero entries of the erasure mask')

        if not self.decodeable(erasure_mask):
            raise DecoderError('Can\'t decode the given erasure pattern.')

        # The number of output rows per block varies due to the
        # erasure mask
        output_index = 0

        # Split the blocks
        output_erasure_list = list()
        for i in range(self.blocks):
            block_erasure = erasure_mask[i * self.block_rows:(i+1) * self.block_rows]
            block_index = np.array([not x for x in block_erasure], dtype=bool)
            output_block = output[output_index:output_index + block_index.sum(), :]
            output_index += block_index.sum()
            output_erasure_list.append((output_block, block_erasure))

        # Decode the blocks in parallel
        with Pool(processes=processes) as pool:
            source_blocks = pool.map(self.decode_block, output_erasure_list)

        # Merge the results
        source = np.zeros([self.cols, output.shape[1]])
        for i in range(self.blocks):
            source_block = source_blocks[i]
            source[i * self.block_cols:(i+1) * self.block_cols, :] = source_block

        return source

def null(matrix, eps=1e-12):
    """ Return a matrix whose columns span the null space of matrix.

    Source:
    https://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
    """
    _, s, vh = np.linalg.svd(matrix)
    padding = max(0, np.shape(matrix)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,), dtype=bool)), axis=0)
    null_space = sp.compress(null_mask, vh, axis=0)
    return sp.transpose(null_space)

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
        bdm, _ = BDMEncoder.construct_bdm(5, 4, 1)
        bdm_null = null(bdm.T)
        self.assertTrue(np.dot(bdm_null.T, bdm).sum() < eps)

        bdm, _ = BDMEncoder.construct_bdm(12, 8, 4)
        bdm_null = null(bdm.T)
        self.assertTrue(np.dot(bdm_null.T, bdm).sum() < eps)

        with self.assertRaises(EncoderError):
            BDMEncoder.construct_bdm(12, 8, 8)

    def test_erasure_pattern(self):
        """ Test that decodeable erasure patterns are detected properly. """
        rows = 16
        cols = 8
        blocks = 4
        encoder = BDMEncoder(rows, cols, blocks)
        erasure_mask = np.array(np.zeros(rows), dtype=bool)
        erasure_mask[0:2] = True
        erasure_mask[-3:-1] = True
        self.assertTrue(encoder.decodeable(erasure_mask))

        erasure_mask[3] = True
        self.assertFalse(encoder.decodeable(erasure_mask))

    def test_encode_decode(self):
        """ Test encoder/decoder """
        # Setup the encoder
        rows = 16
        cols = 8
        blocks = 4
        encoder = BDMEncoder(rows, cols, blocks)

        # Generate random input data and encode it
        source = np.random.randn(cols, 10)
        output = encoder.encode(source)

        # Set some erasures
        erasure_mask = np.array(np.zeros(rows), dtype=bool)
        erasure_mask[0:2] = True
        erasure_mask[-3:-1] = True

        # Remove the corresponding rows of the output matrix
        erased_output = output[np.invert(erasure_mask), :]

        # Decode
        source_est = encoder.decode(erased_output, erasure_mask)
        self.assertTrue(np.allclose(source, source_est))

        # Erase another row
        erasure_mask[3] = True
        erased_output = output[np.invert(erasure_mask), :]
        with self.assertRaises(DecoderError):
            encoder.decode(erased_output, erasure_mask)

if __name__ == '__main__':
    unittest.main()
