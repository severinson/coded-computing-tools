'''This module contains code for computing the complexity of various
operations.

'''

import math
import random
import matplotlib.pyplot as plt
import numpy as np
import stats

# relative cost of encoding and decoding
ADDITION_COMPLEXITY = 8
MULTIPLICATION_COMPLEXITY = 24 # n logn
# ADDITION_COMPLEXITY = 64
# MULTIPLICATION_COMPLEXITY = 64*math.log2(64) # n logn

def partitioned_encode_delay(parameters, partitions=None, algorithm='fft'):
    '''Compute delay incurred by the encoding phase. Assumes a shifted exponential
    distribution.

    Args:

    parameters: System parameters.

    partitions: The number of partitions. If None, the value in parameters is
    used.

    algorithm: 'gen' for generator matrix multiplication, 'bm' for
    Berlekamp-Massey and 'fft' for one based on the fast Fourier
    transform.

    Returns: The reduce delay.

    '''
    assert partitions is None or partitions % 1 == 0
    assert algorithm in ['gen', 'bm', 'fft'], algorithm
    if partitions is None:
        partitions = parameters.num_partitions

    delay = stats.order_mean_shiftexp(
        parameters.num_servers,
        parameters.num_servers,
    )

    # scale by the total encoding complexity
    if algorithm == 'gen':
        delay *= block_diagonal_encoding_complexity(
            parameters,
            partitions=partitions,
        )
        # take into account that each coded row is stored at server_storage*q
        # servers. each coded row is thus encoded several times.
        delay *= parameters.muq
    elif algorithm == 'bm':
        partition_length = parameters.num_coded_rows / partitions
        delay *= block_diagonal_decoding_complexity(
            parameters.num_coded_rows,
            1,
            1 - parameters.q / parameters.num_servers,
            partitions,
        )
        delay *= parameters.num_columns
        delay *= parameters.num_servers
    elif algorithm == 'fft':
        partition_length = parameters.num_coded_rows / partitions
        delay *= rs_decoding_complexity_fft(partition_length)*partitions
        delay *= parameters.num_columns
        delay *= parameters.num_servers
    else:
        raise ValueError('algorithm must be either "gen", "bm" or "fft"')

    # split the work over all servers
    delay /= parameters.num_servers

    return delay

def block_diagonal_encoding_complexity(parameters, partitions=None):
    assert partitions is None or partitions % 1 == 0
    if partitions is None:
        partitions = parameters.num_partitions
    multiplications = parameters.num_source_rows / partitions
    multiplications *= parameters.num_coded_rows * parameters.num_columns
    additions = parameters.num_source_rows / partitions - 1
    additions *= parameters.num_coded_rows * parameters.num_columns
    return MULTIPLICATION_COMPLEXITY * multiplications + ADDITION_COMPLEXITY * additions

def stragglerc_encode_delay(parameters):
    '''Compute reduce delay for a system using only straggler coding, i.e., using
    an erasure code to deal with stragglers but no coded multicasting.

    Args:

    parameters: System parameters.

    Returns: The reduce delay.

    '''
    partitions = parameters.num_source_rows / parameters.q
    return partitioned_encode_delay(parameters, partitions=partitions)

def partitioned_reduce_delay(parameters, partitions=None, algorithm='fft'):
    '''Compute delay incurred by the reduce phase. Assumes a shifted
    exponential distribution.

    Args:

    parameters: System parameters.

    partitions: The number of partitions. If None, the value in
    parameters is used.

    algorithm: 'bm' for Berlekamp-Massey and 'fft' for one based on the fast
    Fourier transform.

    Returns: The reduce delay.

    '''
    assert partitions is None or (isinstance(partitions, int) and partitions > 0)
    assert algorithm in ['bm', 'fft'], algorithm
    if partitions is None:
        partitions = parameters.num_partitions

    delay = stats.order_mean_shiftexp(
        parameters.q,
        parameters.q,
    )

    # scale by the decoding complexity per server
    if algorithm == 'bm':
        delay *= block_diagonal_decoding_complexity(
            parameters.num_coded_rows,
            1,
            1 - parameters.q / parameters.num_servers,
            partitions,
        )
    elif algorithm == 'fft':
        partition_length = parameters.num_coded_rows / partitions
        delay *= rs_decoding_complexity_fft(partition_length)*partitions
    else:
        raise ValueError('algorithm must be either "bm" or "fft"')

    delay *= parameters.num_outputs / parameters.q
    return delay

def partitioned_reduce_complexity(parameters, partitions=None):
    '''reduce complexity per server for the partitioned scheme'''
    if partitions is None:
        partitions = parameters.num_partitions

    complexity = block_diagonal_decoding_complexity(
        parameters.num_coded_rows,
        1,
        1 - parameters.q / parameters.num_servers,
        partitions,
    )
    complexity *= parameters.num_outputs / parameters.q
    return complexity

def stragglerc_reduce_delay(parameters):
    '''Compute reduce delay for a system using only straggler coding,
    i.e., using an erasure code to deal with stragglers but no coded
    multicasting.

    Args:

    parameters: System parameters.

    Returns: The reduce delay.

    '''
    # TODO: Evaluate partitioned_reduce_delay for correct T instead
    delay = stats.order_mean_shiftexp(parameters.q, parameters.q)

    # Scale by decoding complexity
    rows_per_server = parameters.num_source_rows / parameters.q
    delay *= block_diagonal_decoding_complexity(
        parameters.num_servers,
        rows_per_server,
        1 - parameters.q / parameters.num_servers,
        1,
    )
    delay *= parameters.num_outputs / parameters.q
    return delay

def encoding_complexity_from_density(parameters=None, density=None):
    '''compute encoding complexity from the density of the encoding matrix

    args:

    parameters: system parameters

    density: average fraction of non-zero entries in the encoding matrix.

    returns: complexity of the encoding.

    '''
    assert 0 < density <= 1
    multiplications = parameters.num_source_rows * density
    multiplications *= parameters.num_coded_rows * parameters.num_columns
    additions = parameters.num_source_rows * density - 1
    additions *= parameters.num_coded_rows * parameters.num_columns
    return additions * ADDITION_COMPLEXITY + multiplications * MULTIPLICATION_COMPLEXITY

def map_complexity_uncoded(parameters):
    '''uncoded scheme map complexity'''
    server_storage = 1 / parameters.num_servers
    rows_per_server = server_storage * parameters.num_source_rows
    complexity = matrix_vector_complexity(
        rows_per_server,
        parameters.num_columns,
    )
    complexity *= parameters.num_outputs
    return complexity

def map_complexity_cmapred(parameters):
    '''coded MapReduce map complexity'''
    server_storage = parameters.muq / parameters.num_servers
    rows_per_server = server_storage * parameters.num_source_rows
    complexity = matrix_vector_complexity(
        rows_per_server,
        parameters.num_columns,
    )
    complexity *= parameters.num_outputs
    return complexity

def map_complexity_stragglerc(parameters):
    '''straggler coding map complexity'''
    server_storage = 1 / parameters.q
    rows_per_server = server_storage * parameters.num_source_rows
    complexity = matrix_vector_complexity(
        rows_per_server,
        parameters.num_columns,
    )
    complexity *= parameters.num_outputs
    return complexity

def map_complexity_unified(parameters):
    '''unified scheme map complexity'''
    rows_per_server = parameters.server_storage * parameters.num_source_rows
    complexity = matrix_vector_complexity(
        rows_per_server,
        parameters.num_columns,
    )
    complexity *= parameters.num_outputs
    return complexity

def rs_decoding_complexity(code_length, packet_size, erasure_prob):
    '''Compute the decoding complexity of Reed-Solomon codes

    Return the number of operations (additions and multiplications)
    required to decode a Reed-Solomon code over the packet erasure
    channel, and using the Berelkamp-Massey algorithm.

    Args:

    code_length: The length of the code in number of coded symbols.

    packet_size: The size of a packet in number of symbols.

    erasure_prob: The erasure probability of the packet erasure channel.

    Returns: tuple (additions, multiplications)

    '''
    additions = code_length * (erasure_prob * code_length - 1) * packet_size
    multiplications = pow(code_length, 2) * erasure_prob * packet_size;
    return additions * ADDITION_COMPLEXITY + multiplications * MULTIPLICATION_COMPLEXITY

def rs_decoding_complexity_fft(code_length):
    '''Compute the decoding complexity of Reed-Solomon codes

    Return the number of operations (additions and multiplications) required to
    decode a Reed-Solomon code over the packet erasure channel, and using the
    FFT-based algorithm in the paper "Novel Polynomial Basis With Fast Fourier
    Transform and its Application to Reed-Solomon Erasure Codes".

    Parameters are curve-fit to empiric values.

    Args:

    code_length: The length of the code in number of coded symbols.

    Returns: tuple (additions, multiplications)

    '''
    f = lambda x, a, b, c: a+b*x*np.log2(c*x)
    additions = f(code_length, 2, 8.5, 0.86700826)
    multiplications = f(code_length, 2, 1, 4)
    return additions*ADDITION_COMPLEXITY + multiplications*MULTIPLICATION_COMPLEXITY

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
    assert isinstance(packet_size, int) or isinstance(packet_size, float)
    assert isinstance(erasure_prob, float)
    assert isinstance(partitions, int)
    assert code_length % partitions == 0, 'Partitions must divide code_length.'
    partition_length = code_length / partitions
    partition_complexity = rs_decoding_complexity(partition_length, packet_size, erasure_prob)
    return partition_complexity * partitions

def matrix_vector_complexity(rows=None, cols=None):
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
