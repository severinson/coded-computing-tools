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

# This is a simulator of the unified coding scheme presented in
# the article "A Unified Coding Framework for Distributed Computing
# with Straggling Servers", by Songze Li, Mohammad Ali Maddah-Ali, and
# A. Salman Avestimehr.
# The paper is available on arXiv https://arxiv.org/abs/1609.01690,
# and we recommend reading the paper to understand how the scheme
# works.


from scipy.misc import comb as nchoosek
import random
import math
import copy
import itertools as it

# Class representing a single coded symbol
class Symbol(object):
    def __init__(self, partition, index, output):
        assert type(partition) is int
        assert type(index) is int
        assert type(output) is int or output is None
            
        self.partition = partition
        self.index = index
        self.output = output

    def __str__(self):
        #return 'Symbol: {partition: ' + str(self.partition) + ' index: ' + str(self.index) + ' output: ' + str(self.output) + ' }'
        return chr(self.partition + ord('A')) + '_' + str(self.index) + '_' + str(self.output)
        
    # Override default equals behavior
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.partition != other.partition:
                return False

            if self.index != other.index:
                return False

            if self.output != other.output:
                return False

            return True

        return NotImplemented

    # Define not equals test    
    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)

        return NotImplemented

    # Override default hash behavior
    def __hash__(self):
        return hash((self.partition, self.index, self.output))

# Class representing a collection of symbols
class SymbolCollection (object):
    def __init__(self):
        self.symbols = set()

    def __str__(self):
        s = 'SymbolCollection: { '
        for symbol in self.symbols:
            s = s + str(symbol) + ' '

        s = s + ' }'
        return s

    def weight(self):
        return len(self.symbols)

    # Add a symbol or a set of symbols
    def add(self, symbol):        
        assert type(symbol) is Symbol or type(symbol) is set
        if type(symbol) is set:
            for s in symbol:
                assert type(s) is Symbol                
            [self.symbols.add(s) for s in symbol]
        else:
            self.symbols.add(symbol)

    # Peel away symbols from a collection
    def peel(self, symbol_collection):
        # TODO: Optimize
        assert type(symbol_collection) is SymbolCollection

        # Make a copy of the local symbol set
        symbols = self.symbols.copy()

        # Discard all symbols from symbol_collection
        for symbol in symbol_collection.symbols:
            symbols.discard(symbol)

        self.symbols = symbols

    def count(self):
        partitions = dict()
        for s in self.symbols:
            if s.partition not in partitions:
                partitions[s.partition] = 1
            else:
                partitions[s.partition] = partitions[s.partition] + 1

        return partitions

# Class representing a batch containing several symbols and labeled by
# a set of servers
class Batch(object):
    def __init__(self, label):
        assert type(label) is tuple
        
        self.symbols = SymbolCollection()
        self.label = label

    def __str__(self):
        return 'Label:' + str(self.label) + ' ' + str(self.symbols)

    def __repr__(self):
        return self.__str__()

    # Add a symbol to batch
    def add(self, symbol):
        assert type(symbol) is Symbol
        
        # Add symbol to list
        self.symbols.add(symbol);

# Class representing a server
class Server(object):
    def __init__(self, index, k, n, num_partitions, num_outputs):
        self.index = index
        self.needed = SymbolCollection()
        self.symbols = SymbolCollection()
        self.segment = SymbolCollection()
        self.received = list()
        self.outputs = list()
        self.k = k
        self.n = n
        self.num_partitions = num_partitions
        self.num_outputs = num_outputs

    def __str__(self):
        s = 'Server '  + str(self.index) + ': { '
        s = s + 'Stores: ['
        for output in range(self.num_outputs):
            output_symbols = {symbol for symbol in self.symbols.symbols
                              if symbol.output == output}
            if len(output_symbols) == 0:
                continue
            
            s = s + 'output ' + str(output) + ': ['            
            for partition in range(self.num_partitions):
                partition_symbols = {symbol for symbol in output_symbols
                                     if symbol.partition == partition}
                if len(partition_symbols) == 0:
                    continue
                
                s = s + 'partition ' + str(partition) + ': '                
                s = s + str(len(partition_symbols)) + ', '
            s = s + ']'
        s = s + '] '

        s = s + 'Needs: ['
        for output in self.outputs:
            s = s + 'output ' + str(output) + ': ['
            output_symbols = {symbol for symbol in self.needed.symbols
                              if symbol.output == output}
            for partition in range(self.num_partitions):
                s = s + 'partition ' + str(partition) + ': '
                partition_symbols = {symbol for symbol in output_symbols
                                     if symbol.partition == partition}
                s = s + str(len(partition_symbols)) + ', '
            s = s + ']'            
        s = s + '] '        
        s = s + '}'            

        return s

    def __repr__(self):
        s = 'Server ' + str(self.index) + ': { '
        s = s + 'Stores: [' + str(self.symbols) + '] '
        s = s + 'Needs: [' + str(self.needed) + ']'
        s = s + ' }'
        return s

    # Add symbols from a collection
    def add(self, symbol_collection):
        assert type(symbol_collection) is SymbolCollection

        # Add the symbol to set of local symbols and discard it from
        # the needed symbols.
        for symbol in symbol_collection.symbols:
            self.symbols.add(symbol)
            self.needed.symbols.discard(symbol)

        # Discard symbols from outputs and partitions where we hold
        # enough symbols now.
        # TODO: Optimize
        for output in self.outputs:
            output_symbols = {symbol for symbol in self.needed.symbols
                              if symbol.output == output}
            for partition in range(self.num_partitions):
                partition_symbols = {symbol for symbol in output_symbols
                                     if symbol.partition == partition}

                if len(partition_symbols) <= (self.n - self.k):
                    for symbol in partition_symbols:
                        #print('Server', self.index, 'discarded symbol', symbol)
                        self.needed.symbols.discard(symbol)
                        
                
    # Receive a multicasted symbol collection
    def receive(self, symbol_collection):
        assert type(symbol_collection) is SymbolCollection
        
        # Make a copy of the symbol collection to make sure this
        # server owns it.
        collection = copy.deepcopy(symbol_collection)
        
        # Peel away any known symbols
        collection.peel(self.symbols)

        # If all symbols were removed, return
        if collection.weight() == 0:
            return

        # If only 1 symbol remains, add it to the known symbols
        if collection.weight() == 1:
            #print('Server', self.index, 'got a new symbol:', collection)
            self.add(collection)
            return
                            
        # Otherwise add it to the list of symbols to decode later
        #print('Server', self.index, 'couldn\'t decode.')
        self.received.append(collection)

    
# Class representing a computation system
class ComputationSystem(object):
    #def __init__(self, rows_per_batch, num_servers, q, num_outputs, server_storage, num_partitions):
    def __init__(self, parameters):
        assert type(parameters) is SystemParameters
        self.p = parameters
        
        num_coded_rows = parameters.num_coded_rows
        num_partitions = parameters.num_partitions
        num_outputs = parameters.num_outputs
        num_servers = parameters.num_servers
        rows_per_partition = parameters.rows_per_partition

        # Build a list of coded symbols
        symbols = list()
        for i in range(num_coded_rows):
            partition = i % num_partitions
            index = int(i / num_partitions)
            symbol = Symbol(partition, index, 0)
            symbols.append(symbol)

        # Make a set of all symbols for reference
        self.symbols = set()
        for symbol in symbols:
            for output in range(num_outputs):
                self.symbols.add(Symbol(symbol.partition, symbol.index, output))
        
        # Setup servers
        servers = list()
        self.servers = servers
        for server_index in range(num_servers):
            server = Server(server_index, rows_per_partition,
                            num_coded_rows / num_partitions,
                            num_partitions, num_outputs)
            servers.append(server)

    def __str__(self):
        s = 'Computation System:\n'
        for server in self.servers:
            s = s + str(server) + '\n'

        return s

    def __repr__(self):
        s = 'Computation System:\n'
        for server in self.servers:
            s = s + repr(server) + '\n'

        return s
            
    # Store symbols in batches with every batch containing an equal
    # number of symbols per partition.
    def storage_equal_per_batch(self):
        assert self.rows_per_batch / self.num_partitions % 1 == 0, 'There must be an even number of symbols per batch.'

        # Create a separate set of symbols per partition
        symbols_per_partition = list()
        for partition in range(self.num_partitions):
            partition_set = {symbol for symbol in self.symbols
                             if symbol.partition == partition
                             and symbol.output == 0}
            symbols_per_partition.append(partition_set)

        # Make a cycle over the sets
        symbols_per_partition_cycle = it.cycle(symbols_per_partition)
        
        # Store symbols in batches
        batches = set()
        batch_labels = it.combinations(range(self.num_servers), int(self.server_storage * self.q))   
        for batch_index in range(self.num_batches):
            # Label the batch by a unique subset of servers
            label = next(batch_labels)
            batch = Batch(label)
        
            for symbol_index in range(self.rows_per_batch):
                symbol = next(symbols_per_partition_cycle).pop()
                batch.add(symbol)

            batches.add(batch)
        
        self.batches = batches

    # Store symbols in batches with every server containing an equal
    # number of symbols per partition.
    def storage_equal_per_server(self):
        rows_per_partition_per_server = self.p.server_storage * self.p.num_source_rows / self.p.num_partitions
        assert rows_per_partition_per_server  % 1 == 0

        # Create a separate set of symbols per partition
        symbols_per_partition = list()
        for partition in range(self.p.num_partitions):
            partition_set = {symbol for symbol in self.symbols
                             if symbol.partition == partition
                             and symbol.output == 0}
            symbols_per_partition.append(partition_set)

        # Keep track of how many symbols per partition every server
        # holds
        server_partition_count = dict()
        for server_index in range(self.p.num_servers):
            server_partition_count[server_index] = dict()
            for partition in range(self.p.num_partitions):
                server_partition_count[server_index][partition] = 0
            
        labels = it.combinations(range(self.p.num_servers), int(self.p.server_storage * self.p.q))
        for label in labels:
            collection = SymbolCollection()
            for batch_symbol_index in range(self.p.rows_per_batch):
                score = math.inf
                selected_partition = None
                for partition in range(self.p.num_partitions):

                    # Don't evaluate partitions with no symbols left                    
                    if len(symbols_per_partition[partition]) == 0:
                        continue
                    
                    tentative_score = 1 - len(symbols_per_partition[partition]) / (self.p.num_coded_rows / self.p.num_partitions)
                    for server in [self.servers[index] for index in label]:
                        tentative_score = tentative_score + math.pow(server_partition_count[server.index][partition],2)

                    if tentative_score < score:
                        score = tentative_score
                        selected_partition = partition

                # print('Selected partition', selected_partition, 'with score', score, 'for label', label)
                
                # Add a random symbol from the selected partition
                collection.add(symbols_per_partition[selected_partition].pop())

                # Update the partition counts
                for server in [self.servers[index] for index in label]:
                    server_partition_count[server.index][selected_partition] = server_partition_count[server.index][selected_partition] + 1

            # Store the collection on the selected servers
            for server in [self.servers[index] for index in label]:
                server.add(collection)

    # Store symbols randomly
    def storage_random(self):
        symbols = {symbol for symbol in self.symbols if symbol.output == 0}
        
        # Store symbols in batches
        batches = set()
        batch_labels = it.combinations(range(self.num_servers), int(self.server_storage * self.q))   
        for batch_index in range(self.num_batches):
            # Label the batch by a unique subset of servers
            label = next(batch_labels)
            batch = Batch(label)
        
            for symbol_index in range(self.rows_per_batch):
                batch.add(symbols.pop())

            batches.add(batch)

        self.batches = batches

    # Store symbols from an optimized assignment
    def storage_from_assignment(self, X, A):

        # Create a separate set of symbols per partition
        symbols_per_partition = list()
        for partition in range(self.p.num_partitions):
            partition_set = {symbol for symbol in self.symbols
                             if symbol.partition == partition
                             and symbol.output == 0}
            symbols_per_partition.append(partition_set)

        collections = list()
        for row in X:
            collection = SymbolCollection()
            for col in range(X.shape[1]):
                [collection.add(symbols_per_partition[col].pop()) for x in range(int(row[col]))]
            collections.append(collection)

        for server_index in range(len(A)):
            server = self.servers[server_index]            
            for batch_index in A[server_index]:
                server.add(collections[batch_index])

    # Add the set of batches to servers
    def assign_batches(self):
        for batch in self.batches:
            for server_index in batch.label:
                server = self.servers[server_index]
                server.add(batch.symbols)

    def storage_eval(self, subsets = None):
        print('Evaluating storage design...')

        if subsets is None:
            subsets = it.combinations(range(self.num_servers), int(self.q * self.server_storage))
            
        for subset in subsets:
            servers = [self.servers[index] for index in subset]
            symbol_sets = [server.symbols.symbols for server in servers]
            symbols = set.intersection(*symbol_sets)
            #[symbols.add(x) for symbol_set in symbol_sets for x in symbol_set if x not in symbols]
            collection = SymbolCollection()
            collection.symbols = symbols
            print('Servers', subset, 'share symbols', collection)
        
    # Verify that the storage design is feasible in the sense that any
    # q servers hold enough rows to decode all partitions.    
    def verify(self):
        print('Verifying storage scheme...')
        valid = True                    
        
        # Construct an iterator of all subsets of q servers
        server_subsets = it.combinations(range(self.num_servers), self.q)

        # Iterate over the possible cardinality q subsets
        for subset in server_subsets:
            print('Checking subset', subset)

            # The set of unique symbols among the q servers
            symbols = set()

            # Iterate over the q servers and take the set union
            servers = [self.servers[index] for index in subset]
            for server in servers:
                symbols = symbols | server.symbols.symbols

            unique_symbols = len(symbols)
            if unique_symbols < self.num_source_rows:
                print('ERROR: Storage holds', unique_symbols, 'unique rows, but need', self.num_source_rows)
                valid = False
                break

            print('Storage holds', unique_symbols, 'unique symbols.')

            # Count the number of symbols per partition
            for partition in range(self.num_partitions):
                partition_symbols = [symbol for symbol in symbols if symbol.partition == partition]
                unique_symbols_partition = len(partition_symbols)
                print(unique_symbols_partition, 'unique symbols for partition', partition)
                if unique_symbols_partition < (self.num_source_rows / self.num_partitions):
                    print('ERROR: Storage holds', unique_symbols_partition,
                          'for partition', partition, 'but needs', self.num_source_rows / self.num_partitions)
                    valid = False
                    break

            print('')
            # TODO: Should we always break after testing one subset?
            #break
            
        print('Storage is valid:', valid)

    # Perform map phase
    def map(self, subset=None):
        # Select q random servers unless one was provided
        if subset is None:
            subset = random.choice(list(it.combinations(range(self.p.num_servers), self.p.q)))
        
        print('Selected subset', subset, 'in the map phase')

        outputs_per_server = int(self.p.num_outputs / self.p.q)

        # Make a list of the finished servers
        self.finished_servers = [self.servers[index] for index in subset]

        # Assign outputs to the servers
        # TODO: Find a more pythonic way of doing this
        output_index = 0
        for server in self.finished_servers:
            print(server)
            for i in range(outputs_per_server):
                server.outputs.append(output_index)
                output_index = output_index + 1

        # Update the sets of symbols the servers hold to include outputs
        for server in self.finished_servers:
            output_symbols = SymbolCollection()
            for symbol in server.symbols.symbols:
                for output in range(self.p.num_outputs):                    
                    output_symbols.add(Symbol(symbol.partition, symbol.index, output))
            server.symbols = output_symbols

        # Assign the set of symbols the servers still need
        for server in self.finished_servers:
            server.needed.symbols = {symbol for symbol in self.symbols
                                     if symbol not in server.symbols.symbols
                                     and symbol.output in server.outputs}
            
            # Remove the partitions where the server hold enough
            # symbols already
            for partition in range(self.p.num_partitions):
                partition_symbols = {symbol for symbol in server.needed.symbols if symbol.partition == partition}
                if len(partition_symbols) <= (self.p.num_coded_rows - self.p.num_source_rows) / self.p.num_partitions:
                    server.needed.symbols.discard(partition_symbols)

        return subset

    # Perform shuffle phase
    def shuffle(self):
        print('Performing shuffle phase...')
        
        # Count the transmissions
        self.transmissions = 0

        # Make a set to be populated by all symbols part of any
        # multicast iteration.
        psi = dict()
        for server in self.finished_servers:
            psi[server.index] = SymbolCollection()

        multicast_index = int(self.p.server_storage * self.p.q) # j in Li2016
        while multicast_index > 1:
            li_transmissions = self.li_multicast(multicast_index, psi)
            self.transmissions = self.transmissions + li_transmissions
            multicast_index = multicast_index - 1

            print('Li multicast iteration ended after', li_transmissions, 'transmissions.')

        uni_transmissions = self.unicast()
        self.transmissions = self.transmissions + uni_transmissions

        communication_load = self.transmissions / self.p.num_source_rows / self.p.num_outputs
        print('Unicast phase ended after', uni_transmissions, 'transmission.')
        print('Shuffle phase ended after', self.transmissions,
              'transmissions with a load of', communication_load)

        return communication_load, psi

    # Perform a Li multicasting iteration as described in the paper by
    # Li et al. arXiv:1609.01690v1
    def li_multicast(self, multicast_index, psi):
        transmissions = 0
        multicasting_subsets = it.combinations(range(self.p.q), multicast_index + 1) # S in li 0216
        num_subsets = nchoosek(self.p.q, multicast_index + 1)
        subset_index = 0

        # Record the number of symbols each server holds
        num_symbols = dict()
        for server in self.finished_servers:
            num_symbols[server.index] = len(server.symbols.symbols)

        print('Initiating Li multicast with index', multicast_index, 'and', num_subsets, 'subsets.')
        for subset in multicasting_subsets:
            #print('subset:', [self.p.finished_servers[i].index for i in subset])

            # Print the progress
            subset_index = subset_index + 1
            if subset_index % 100 == 0:
                print(str(int(subset_index / num_subsets * 100)) + '%')

            for k in range(len(subset)):
                #print('k=', k)
                #print('servers_without_k=', subset[0:k] + subset[k+1:])
                
                server_k = self.finished_servers[subset[k]]
                #servers_without_k = [self.p.finished_servers[index] for index in subset[0:k] + subset[k+1:]]
                servers_without_k = [self.finished_servers[index] for index in subset if index != subset[k]]

                # The symbols needed by k
                nu = {symbol for symbol in server_k.needed.symbols}

                # Intersected by the set of symbols known by the others
                for server in servers_without_k:
                    nu = nu & server.symbols.symbols

                #print('k=', server_k.index, '|nu|=', len(nu))

                # Add symbols to psi. psi is the set of all symbols
                # part of any multicast iteration. Kept for analytical
                # purposes.
                psi[server_k.index].add(set.intersection(*[server.symbols.symbols for server in servers_without_k]))

                # Split nu evenly over servers_without_k
                server_cycle = it.cycle(servers_without_k)
                while len(nu) > 0:
                     server = next(server_cycle)
                     server.segment.add(nu.pop())

            #print('---- Multicasting Round ----')                
            for server in [self.finished_servers[index] for index in subset]:

                # Transmit as long as there are symbols left in this
                # server's segment
                while len(server.segment.symbols) > 0:
                
                    # Select one symbol for every receiving server
                    collection = SymbolCollection()
                    for receiving_server in [self.finished_servers[index] for index in subset]:
                        if receiving_server.index == server.index:
                            continue

                        symbols = {symbol for symbol in server.segment.symbols
                                   if symbol in receiving_server.needed.symbols
                                   and symbol not in collection.symbols}

                        # Continue if there were no overlapping symbols
                        if len(symbols) == 0:
                            continue

                        symbol = symbols.pop()
                        #print('symbol=', symbol)
                        collection.add(symbol)
                        server.segment.symbols.remove(symbol)

                    # Break if we found no multicasting opportunity.
                    if len(collection.symbols) < 1:                        
                        break

                    # Multicast collection                    
                    self.multicast(server.index, collection)
                    transmissions = transmissions + 1

                # Reset segment
                server.segment = SymbolCollection()        

        for server in self.finished_servers:
            print('Server', server.index, 'recovered',
                  len(server.symbols.symbols) - num_symbols[server.index], 'symbols.')

        rec_bound = nchoosek(self.p.q - 1, multicast_index) * nchoosek(self.p.num_servers - self.p.q, int(self.p.server_storage * self.p.q) - multicast_index) * self.p.num_source_rows / (self.p.q / self.p.num_servers * nchoosek(self.p.num_servers, int(self.p.server_storage * self.p.q)))

        print('Bound is:', rec_bound)
            
        return transmissions

    # Multicast a symbol from one server to the others
    def multicast(self, source, symbol_collection):
        #print('Multicast', symbol_collection, 'from server', source)
        receveing_servers = [server for server in self.finished_servers if server.index != source]
        for server in receveing_servers:
            server.receive(symbol_collection)

    # Unicast until every server holds enough symbols
    def unicast(self):
        print('Unicasting until every server hold enough symbols...')
        transmissions = 0
        for server in self.finished_servers:
            for receiving_server in self.finished_servers:

                # Don't transmit to yourself
                if receiving_server.index == server.index:
                    continue

                #symbols = server.symbols.symbols & receiving_server.needed.symbols
                while True:
                    symbols = {s for s in server.symbols.symbols if s in receiving_server.needed.symbols}
                    if len(symbols) == 0:
                        break
                    symbol_collection = SymbolCollection()
                    symbol_collection.add(symbols.pop())
                    receiving_server.receive(symbol_collection)
                    transmissions = transmissions + 1

        return transmissions            
