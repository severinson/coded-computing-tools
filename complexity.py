import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Source: http://web.eecs.utk.edu/~plank/plank/papers/FAST-2005.pdf
LDPC_RATE = 1.0496
LDPC_XOR_PER_WORD = 30 / 4
RS_XOR_PER_WORD = 44.6


# Computational complexity of arithmetic operations
# Source:
# https://streamcomputing.eu/blog/2012-07-16/how-expensive-is-an-operation-on-a-cpu/
ADDITION_COMPLEXITY = 1
MULTIPLICATION_COMPLEXITY = 4

def rs_decoding_complexity(code_length, packet_size, erasure_prob):
    '''Compute the decoding complexity of Reed-Solomon codes

    Return the number of operations (additions and multiplications)
    required to decode a Reed-Solomon code over the packet erasure
    channel, and using the Berelkamp-Massey algorithm.

    Args:
    code_length: The length of the code in number of coded symbols.
    packet_size: The size of a packet in number of symbols.
    erasure_prob: The erasure probability of the packet erasure channel.

    Returns:
    The number of operations (additions and multiplications)
    required to decode.

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
    The number of operations (additions and multiplications)
    required to decode.

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

    Returns:
    The number of operations (additions and multiplications)
    required to decode.

    '''
    assert isinstance(code_length, int)
    assert isinstance(packet_size, int)
    assert isinstance(erasure_prob, float)
    assert isinstance(partitions, int)
    assert code_length % partitions == 0, 'Partitions must divide code_length.'
    partition_length = code_length / partitions

    # TODO: Will this compute the correct complexity? There might be a
    # problem in that a packet may contain symbols from various partitions.
    partition_complexity = rs_decoding_complexity(partition_length, packet_size, erasure_prob)
    return partition_complexity * partitions

def matrix_vector_complexity(rows, cols):
    '''Compute the complexity of matrix-vector multiplication

    Return the complexity of multiplying a matrix A with number of
    rows and columns as given in the argument by a vector x. The
    multiplication is done as A*x.

    Args:
    rows: The number of rows of the matrix.
    cols: The number of columns of the matrix.

    Returns:
    The complexity of the multiplication.

    '''
    additions = cols * rows
    multiplications = cols * rows
    return additions * ADDITION_COMPLEXITY + multiplications * MULTIPLICATION_COMPLEXITY

def order_stat(icdf, total, kth):
    '''Compute the order statistic of a given distribution numerically

    Compute the k:th order statistic for total number of random
    variables with distribution given by the argument. Runs in O(total).

    Args:
    icdf: The inverse cumulative distribution function of the
    probability distribution of interest. Provided as a function with
    one argument.
    total: The total number of realizations.
    kth: The order to compute the statistic of.

    Returns:
    The k:th order statistic for total number of random variables
    with distribution given by the argument.

    '''
    # assert isinstance(icdf, fun)
    assert isinstance(total, int)
    assert isinstance(kth, int)
    assert total >= kth, 'kth must be less or equal to total.'
    realizations = [icdf(random.random()) for _ in range(total)]
    return np.partition(np.asarray(realizations), kth)[kth]

def expected_order_stat(icdf, total, kth, samples=1000):
    '''Compute the expected order statistic numerically

    Args:
    icdf: The inverse cumulative distribution function of the
    probability distribution of interest. Provided as a function with
    one argument.
    total: The total number of realizations.
    kth: The order to compute the statistic of.
    samples: Number of times to sample the order statistic.

    Returns:
    The expected k:th order statistic for total number of random variables
    with distribution given by the argument.
    '''
    realizations = [order_stat(icdf, total, kth) for _ in range(samples)]
    return sum(realizations) / (len(realizations) + 1)

def order_mean_shifted_exponential(total, kth, mu=1):
    '''Compute the expected k:th order statistic of a shifted exponential distribution

    The expectation is computed analytically.

    Args:
    total: The total number of realizations.
    kth: The order to compute the statistic of.
    mu: The exponential distribution parameter. Also called the straggling parameter.

    Returns:
    The k:th order statistic for total number of random variables
    distributed accoridng to the shifted expinential distribution.
    '''
    return 1 + (1 / mu) * math.log(total / (total - kth))

def icdf_shifted_exponential(value, mu=1):
    '''ICDF of the shifted exponential distribution.

    Pass uniformly distributed random numbers into this function to
    generate random numbers from the shifted exponential distribution.

    Args:
    value: The value to compute the ICDF of. Must be less than 1.
    mu: The exponential distribution parameter. Also called the straggling parameter.

    Returns:
    The ICDF evaluated at value.

    '''
    assert value < 1, 'Cannot compute the ICDF of values larger than or equal to 1.'
    return 1 - math.log(1 - value) / mu;

def main():
    map_servers = 600
    map_threshold = 400
    rate = map_threshold / map_servers

    # Compute the map latency by scaling it by the number of
    # operations performed by each server.

    rows = 1e5
    cols = 1e5
    vevtors_per_server = map_threshold

    map_complexity = matrix_vector_complexity(rows, cols, vectors )

    samples = 500
    map_runtimes = [order_stat(icdf_shifted_exponential, k, q) for _ in range(samples)]
    sample_mean = sum(runtimes) / (len(runtimes) + 1)

    # analytical_mean = order_mean_shifted_exponential(k, q)
    # print('Sample mean: ' + str(sample_mean) + '. Analytical mean: ' + str(analytical_mean) + '.')

    fig = plt.figure()
    plt.hist(runtimes)
    plt.grid(True, which='both')
    plt.ylabel('Probability', fontsize=18)
    plt.xlabel('Runtime', fontsize=18)
    plt.autoscale()
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    main()
