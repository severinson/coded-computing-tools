############################################################################
# Copyright 2016 Albin Severinson                                          #
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

'''This module provides functionality for exhaustively evaluating the
performance of an assignment. Specifically, the performance is
evaluated by evaluating all possible orders in which servers can
finish. This is only possible for a very small number of servers.

'''

def communication_load(par, assignment_matrix, labels, Q=None):
    '''Calculate the communication load of an assignment.

    Count the number of unicasts required for all servers to hold
    enough symbols to deocde all partitions for some assignment
    exhaustively.

    Args:
    par: System parameters

    assignment_matrix: Assignment matrix

    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.

    Q: Leave at None to evaluate all subsets Q, or set to a tuple of servers to
    count unicasts only for some specific Q.

    Returns: Tuple with the first element the average number of
    unicasts required and second element the worst case number of
    unicasts.

    '''

    raise NotImplementedError('This function has not yet been patched to work with the sparse assignments. Use the sampled version instead.')

    # Count the total and worst-case score
    total_score = 0
    worst_score = 0

    # If a specific Q was given evaluate only that one.  Otherwise
    # evaluate all possible Q.
    if Q is None:
        finished_servers_subsets = it.combinations(range(par.num_servers), par.q)
        num_subsets = nchoosek(par.num_servers, par.q)
    else:
        finished_servers_subsets = [Q]
        num_subsets = 1

    for finished_servers in finished_servers_subsets:
        set_score = 0

        # Count over all server perspectives
        for server in finished_servers:
            # Create the set of all symbols stored at k or sent to k
            # via multicast.
            rows = set()

            for batch in labels[server]:
                rows.add(batch)

            for j in range(par.sq, int(par.server_storage*par.q) + 1):
                for multicast_subset in it.combinations([x for x in finished_servers if x != server], j):
                    rows = rows | set.intersection(*[labels[x] for x in multicast_subset])

            selector_vector = np.zeros(par.num_batches)
            for row in rows:
                selector_vector[row] = 1

            count_vector = np.dot(selector_vector, assignment_matrix)
            count_vector -= par.num_source_rows / par.num_partitions
            score = model.remaining_unicasts(count_vector)

            total_score = total_score + score
            set_score = set_score + score

        if set_score > worst_score:
            worst_score = set_score

    average_score = total_score / num_subsets
    return average_score, worst_score

def computational_delay(par, assignment_matrix, labels, Q=None):
    '''Calculate the computational delay of an assignment.

    Calculate the number of servers required to decode all partitions
    for an assignment exhaustively.

    Args:

    par: System parameters

    assignment_matrix: Assignment matrix

    labels: List of sets, where the elements of the set labels[i] are
    the rows of the assignment_matrix stored at server i.

    Returns: Number of servers required averaged over all subsets.

    '''

    raise NotImplementedError('There is a bug in this function. When computing the delay, it will only add at most one additional server over q. Use the sampled version instead.')

    # Count the total and worst-case score
    total_score = 0
    worst_score = 0

    # If a specific Q was given evaluate only that one.  Otherwise
    # evaluate all possible Q.
    if Q is None:
        finished_servers_subsets = it.combinations(range(par.num_servers), par.q)
        num_subsets = nchoosek(par.num_servers, par.q)
    else:
        finished_servers_subsets = [Q]
        num_subsets = 1

    for finished_servers in finished_servers_subsets:
        set_score = 0

        # Count the total number of symbols per partition
        rows_by_partition = np.zeros(par.num_partitions)

        # Keep track of the batches we've added
        batches = set()

        # Add the batches from the first q servers
        for server in finished_servers:
            for batch in labels[server]:
                batches.add(batch)

        for batch in batches:
            rows_by_partition = np.add(rows_by_partition, assignment_matrix[batch])
        set_score = par.q

        # TODO: Add more servers until all partitions can be decoded
        if not enough_symbols(par, rows_by_partition):
            set_score += 1

        delay = par.computational_delay(q=set_score)

        total_score += delay
        if delay > worst_score:
            worst_score = delay

    average_score = total_score / num_subsets
    return average_score, worst_score
