'''This module contains code for computing the complexity of various
operations.

'''

import math
import random
import matplotlib.pyplot as plt
import numpy as np
import stats
import lt

# Source: http://web.eecs.utk.edu/~plank/plank/papers/FAST-2005.pdf
LDPC_RATE = 1.0496
LDPC_XOR_PER_WORD = 30 / 4
RS_XOR_PER_WORD = 44.6

# Finite field arithmetic complexity for fields of size 2^100
# Source: Zhi, L., Higgins, J., & Clement, M. (2001).
# Performance of finite field arithmetic in an elliptic curve cryptosystem.

# Ignore addition cost to simplify analysis
# ADDITION_COMPLEXITY = 0.13e-6
# ADDITION_COMPLEXITY = 0
# MULTIPLICATION_COMPLEXITY = 87e-6
ADDITION_COMPLEXITY = 0
MULTIPLICATION_COMPLEXITY = 1

def partitioned_reduce_delay(parameters, partitions=None):
    '''Compute delay incurred by the reduce phase. Assumes a shifted
    exponential distribution.

    Args:

    parameters: System parameters.

    partitions: The number of partitions. If None, the value in
    parameters is used.

    Returns: The reduce delay.

    '''
    assert partitions is None or (isinstance(partitions, int) and partitions > 0)
    if partitions is None:
        partitions = parameters.num_partitions

    delay = stats.order_mean_shiftexp(parameters.q, parameters.q)

    # Scale by decoding complexity
    delay *= block_diagonal_decoding_complexity(parameters.num_coded_rows,
                                                1, 1 - parameters.q / parameters.num_servers,
                                                partitions)
    delay *= parameters.num_outputs / parameters.q
    return delay

def lt_reduce_delay(parameters):
    '''Compute delay incurred by the reduce phase using an LT code.
    Assumes a shifted exponential distribution.

    Args:

    parameters: System parameters.

    Returns: The reduce delay.

    '''
    delay = stats.order_mean_shiftexp(parameters.q, parameters.q)

    # Scale by decoding complexity
    delay *= peeling_decoding_complexity(parameters.num_source_rows)
    delay *= parameters.num_outputs / parameters.q
    return delay

def rs_decoding_complexity(code_length, packet_size, erasure_prob):
    '''Compute the decoding complexity of Reed-Solomon codes

    Return the number of operations (additions and multiplications)
    required to decode a Reed-Solomon code over the packet erasure
    channel, and using the Berelkamp-Massey algorithm.

    Args:
    code_length: The length of the code in number of coded symbols.
    packet_size: The size of a packet in number of symbols.
    erasure_prob: The erasure probability of the packet erasure channel.

    Returns: The total complexity of decoding.

    '''
    additions = code_length * (erasure_prob * code_length - 1) * packet_size
    multiplications = pow(code_length, 2) * erasure_prob * packet_size;
    return additions * ADDITION_COMPLEXITY + multiplications * MULTIPLICATION_COMPLEXITY

def rs_decoding_complexity_large_packets(code_length, erasure_prob):
    '''Compute the decoding complexity of Reed-Solomon codes

    Return the number of operations (additions and multiplications)
    required to decode a Reed-Solomon code over the packet erasure
    channel, and using the Berelkamp-Massey algorithm. This function
    considers the asymptotic case as the packet size approaches
    infinity.

    Args:
    code_length: The length of the code in number of coded symbols.
    erasure_prob: The erasure probability of the packet erasure channel.

    Returns:
    The total complexity of decoding.

    '''
    raise NotImplemented('Use the rs_decoding_complexity() method instead.')
    additions = code_length * (erasure_prob * code_length - 1)
    multiplications = pow(code_length, 2) * erasure_prob
    return additions * ADDITION_COMPLEXITY + multiplications * MULTIPLICATION_COMPLEXITY

def block_diagonal_decoding_complexity(code_length, packet_size, erasure_prob, partitions):
    '''Compute the decoding complexity of a block-diagonal code

    Return the number of operations (additions and multiplications)
    required to decode a block-diagonal code over the packet erasure
    channel, and using the Berelkamp-Massey algorithm. This function
    considers the asymptotic case as the packet size approaches
    infinity.

    Args:
    code_length: The length of the code in number of coded symbols.
    packet_size: The size of a packet in number of symbols.
    erasure_prob: The erasure probability of the packet erasure channel.
    partitions: The number of block-diagonal code partitions.

    Returns: The total complexity of decoding.

    '''
    assert isinstance(code_length, int)
    assert isinstance(packet_size, int)
    assert isinstance(erasure_prob, float)
    assert isinstance(partitions, int)
    assert code_length % partitions == 0, 'Partitions must divide code_length.'
    partition_length = code_length / partitions
    partition_complexity = rs_decoding_complexity(partition_length, packet_size, erasure_prob)
    return partition_complexity * partitions

def peeling_decoding_complexity(source_length):
    '''Compute the decoding complexity of an LT erasure code using a
    peeling decoder.

    Args:

    source_length: Number of source symbols.

    '''
    lt_simulation = lt.lt_overhead(source_length)
    return lt_simulation['additions'] * ADDITION_COMPLEXITY

def peeling_decoding_complexity_old(code_length, code_rate, erasure_prob):

    '''Compute the decoding complexity of decoding an erasure code using a
    peeling decoder.

    Args:

    code_length: The length of the code in number of coded symbols.

    code_rate: The rate of the code as given by
    num_information_symbols / num_coded_symbols.

    erasure_prob: The packet erasure probability.

    Returns: The total complexity of decoding.

    '''
    information_symbols = round(code_length * code_rate)

    # Compute the average degree for the ideal Soliton distribution
    avg_check_degree = 1 / information_symbols
    avg_check_degree += sum([k / (k * (k - 1)) for k in range(2, information_symbols + 1)])

    num_iterations = round(code_length * erasure_prob)
    additions_per_iteration = avg_check_degree - 1
    multiplications_per_iteration = avg_check_degree
    complexity_per_iteration = additions_per_iteration * ADDITION_COMPLEXITY
    complexity_per_iteration += multiplications_per_iteration * MULTIPLICATION_COMPLEXITY
    return num_iterations * complexity_per_iteration

def matrix_vector_complexity(rows, cols):
    '''Compute the complexity of matrix-vector multiplication

    Return the complexity of multiplying a matrix A with number of
    rows and columns as given in the argument by a vector x. The
    multiplication is done as A*x.

    Args:

    rows: The number of rows of the matrix.

    cols: The number of columns of the matrix.

    Returns: The complexity of the multiplication.

    '''
    additions = cols * rows - 1
    multiplications = cols * rows
    return additions * ADDITION_COMPLEXITY + multiplications * MULTIPLICATION_COMPLEXITY
