import numpy as np
import scipy as sp
import scipy.misc
from scipy.misc import comb as nchoosek
import itertools as it
import copy
import random
import math

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

# System parameters goes here        
class SystemParameters(object):
    def __init__(self, rows_per_batch, num_servers, q, num_outputs, server_storage, num_partitions):
        # TODO: Raise exceptions instead of assertions        
        assert type(rows_per_batch) is int
        assert type(num_servers) is int
        assert type(q) is int
        assert type(num_outputs) is int
        assert type(server_storage) is float or server_storage == 1
        assert type(num_partitions) is int
        
        self.num_servers = num_servers
        self.q = q
        self.num_outputs = num_outputs
        self.server_storage = server_storage
        self.num_partitions = num_partitions
        self.rows_per_batch = rows_per_batch

        assert num_outputs / q % 1 == 0, 'num_outputs must be divisible by the number of servers.'
        assert server_storage * q % 1 == 0, 'server_storage * q must be integer.'
        
        num_batches = scipy.misc.comb(num_servers, server_storage*q)
        assert num_batches % 1 == 0, 'There must be an integer number of batches.'
        num_batches = int(num_batches)
        self.num_batches = num_batches

        num_coded_rows = rows_per_batch * num_batches
        self.num_coded_rows = num_coded_rows

        num_source_rows = q / num_servers * num_coded_rows
        assert num_source_rows % 1 == 0, 'There must be an integer number of source rows.'
        num_source_rows = int(num_source_rows)
        self.num_source_rows = num_source_rows

        assert num_source_rows / num_partitions % 1 == 0, 'There must be an integer number of source rows per partition.'
        rows_per_partition = int(num_source_rows / num_partitions)
        self.rows_per_partition = rows_per_partition
    
# Class representing a computation system
class ComputationSystem(object):
    def __init__(self, rows_per_batch, num_servers, q, num_outputs, server_storage, num_partitions):
        assert type(rows_per_batch) is int
        assert type(num_servers) is int
        assert type(q) is int
        assert type(num_outputs) is int
        assert type(server_storage) is float or server_storage == 1
        assert type(num_partitions) is int
        
        self.num_servers = num_servers
        self.q = q
        self.num_outputs = num_outputs
        self.server_storage = server_storage
        self.num_partitions = num_partitions
        self.rows_per_batch = rows_per_batch

        assert num_outputs / q % 1 == 0, 'num_outputs must be divisible by the number of servers.'
        assert server_storage * q % 1 == 0, 'server_storage * q must be integer.'
        
        num_batches = scipy.misc.comb(num_servers, server_storage*q)
        assert num_batches % 1 == 0, 'There must be an integer number of batches.'
        num_batches = int(num_batches)
        self.num_batches = num_batches

        num_coded_rows = rows_per_batch * num_batches
        self.num_coded_rows = num_coded_rows

        num_source_rows = q / num_servers * num_coded_rows
        assert num_source_rows % 1 == 0, 'There must be an integer number of source rows.'
        num_source_rows = int(num_source_rows)
        self.num_source_rows = num_source_rows

        assert num_source_rows / num_partitions % 1 == 0, 'There must be an integer number of source rows per partition.'
        rows_per_partition = int(num_source_rows / num_partitions)
        self.rows_per_partition = rows_per_partition

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

        '''
        for batch in batches:
            for partition in range(self.num_partitions):
                foo = len({symbol for symbol in batch.symbols.symbols if symbol.partition == partition})
                print('Batch:', batch.label, 'Partition:', partition, '#', foo)
            print('')
        '''
        
        self.batches = batches

    # Store symbols in batches with every server containing an equal
    # number of symbols per partition.
    def storage_equal_per_server(self):
        rows_per_partition_per_server = self.server_storage * self.num_source_rows / self.num_partitions
        assert rows_per_partition_per_server  % 1 == 0

        # Create a separate set of symbols per partition
        symbols_per_partition = list()
        for partition in range(self.num_partitions):
            partition_set = {symbol for symbol in self.symbols
                             if symbol.partition == partition
                             and symbol.output == 0}
            symbols_per_partition.append(partition_set)

        # Keep track of how many symbols per partition every server
        # holds
        server_partition_count = dict()
        for server_index in range(self.num_servers):
            server_partition_count[server_index] = dict()
            for partition in range(self.num_partitions):
                server_partition_count[server_index][partition] = 0
            
        labels = it.combinations(range(self.num_servers), int(self.server_storage * self.q))
        for label in labels:
            collection = SymbolCollection()
            for batch_symbol_index in range(self.rows_per_batch):
                score = math.inf
                selected_partition = None
                for partition in range(self.num_partitions):

                    # Don't evaluate partitions with no symbols left                    
                    if len(symbols_per_partition[partition]) == 0:
                        continue
                    
                    tentative_score = 1 - len(symbols_per_partition[partition]) / (self.num_coded_rows / self.num_partitions)
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

    # Store symbols in batches with every server containing an equal
    # number of symbols per partition.
    def storage_equal_per_server_old(self):
        assert (self.server_storage * self.num_source_rows) / self.num_partitions % 1 == 0

        # Create a separate set of symbols per partition
        symbols_per_partition = list()
        for partition in range(self.num_partitions):
            partition_set = {symbol for symbol in self.symbols
                             if symbol.partition == partition
                             and symbol.output == 0}
            symbols_per_partition.append(partition_set)

        # Make a cycle over the sets
        symbols_per_partition_cycle = it.cycle(symbols_per_partition)
        
        batch_labels = set(it.combinations(range(self.num_servers), int(self.server_storage * self.q)))
        while len(batch_labels) > 0:

            # Create a batch of symbols
            collection = SymbolCollection()
            for index in range(self.rows_per_batch):
                symbol = next(symbols_per_partition_cycle).pop()
                collection.add(symbol)

            # Find a server to store it at
            for label in batch_labels:
                for server_index in label:
                    print('hej')
            
        for label in batch_labels:
            collection = SymbolCollection()
            for index in range(self.rows_per_batch):
                symbol = next(symbols_per_partition_cycle).pop()
                collection.add(symbol)
                
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

    # Add the set of batches to servers
    def assign_batches(self):
        for batch in self.batches:
            for server_index in batch.label:
                server = self.servers[server_index]
                server.add(batch.symbols)

    def storage_eval(self, subsets = None):
        print('Evaluating storage design...')

        if subsets == None:
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
        # Select q random servers
        if subset == None:
            subset = random.choice(list(it.combinations(range(self.num_servers), self.q)))
        
        print('Selected subset', subset, 'in the map phase')

        outputs_per_server = int(self.num_outputs / self.q)

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
                for output in range(self.num_outputs):                    
                    output_symbols.add(Symbol(symbol.partition, symbol.index, output))
            server.symbols = output_symbols

        # Assign the set of symbols the servers still need
        for server in self.finished_servers:
            server.needed.symbols = {symbol for symbol in self.symbols
                                     if symbol not in server.symbols.symbols
                                     and symbol.output in server.outputs}
            
            # Remove the partitions where the server hold enough
            # symbols already
            for partition in range(self.num_partitions):
                partition_symbols = {symbol for symbol in server.needed.symbols if symbol.partition == partition}
                if len(partition_symbols) <= (self.num_coded_rows - self.num_source_rows) / self.num_partitions:
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

        multicast_index = int(self.server_storage * self.q) # j in Li2016
        while multicast_index > 1:
            li_transmissions = self.li_multicast(multicast_index, psi)
            self.transmissions = self.transmissions + li_transmissions
            multicast_index = multicast_index - 1

            print('Li multicast iteration ended after', li_transmissions, 'transmissions.')

        uni_transmissions = self.unicast()
        self.transmissions = self.transmissions + uni_transmissions

        communication_load = self.transmissions / self.num_source_rows / self.num_outputs
        print('Unicast phase ended after', uni_transmissions, 'transmission.')
        print('Shuffle phase ended after', self.transmissions,
              'transmissions with a load of', communication_load)

        return communication_load, psi

    # Perform a Li multicasting iteration as described in the paper by
    # Li et al. arXiv:1609.01690v1
    def li_multicast(self, multicast_index, psi):
        transmissions = 0
        multicasting_subsets = it.combinations(range(self.q), multicast_index + 1) # S in li 0216
        num_subsets = scipy.misc.comb(self.q, multicast_index + 1)
        subset_index = 0

        # Record the number of symbols each server holds
        num_symbols = dict()
        for server in self.finished_servers:
            num_symbols[server.index] = len(server.symbols.symbols)

        #print('Initiating Li multicast with index', multicast_index, 'and', num_subsets, 'subsets.')
        for subset in multicasting_subsets:
            #print('subset:', [self.finished_servers[i].index for i in subset])

            # Print the progress
            subset_index = subset_index + 1
            if subset_index % 100 == 0:
                print(str(int(subset_index / num_subsets * 100)) + '%')

            for k in range(len(subset)):
                #print('k=', k)
                #print('servers_without_k=', subset[0:k] + subset[k+1:])
                
                server_k = self.finished_servers[subset[k]]
                #servers_without_k = [self.finished_servers[index] for index in subset[0:k] + subset[k+1:]]
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

        rec_bound = scipy.misc.comb(self.q - 1, multicast_index) * scipy.misc.comb(self.num_servers - self.q, int(self.server_storage * self.q) - multicast_index) * self.num_source_rows / (self.q / self.num_servers * scipy.misc.comb(self.num_servers, int(self.server_storage * self.q)))

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
    
            
def main():
    rows_per_batch = 2
    num_servers = 6 # K in Li2016
    q = 4
    num_outputs = q # N in Li2016
    server_storage = 1/2 # \mu in Li2016    
    
    num_batches = int(scipy.misc.comb(num_servers, server_storage*q))
    num_coded_rows = rows_per_batch * num_batches
    num_source_rows = q / num_servers * num_coded_rows
    assert num_source_rows % 1 == 0
    num_source_rows = int(num_source_rows)
    
    #num_partitions = num_source_rows*server_storage / 16
    #num_partitions = rows_per_batch
    num_partitions = 10
    assert num_partitions % 1 == 0
    num_partitions = int(num_partitions)

    print('Starting simulation:')    
    print('-------------------')

    '''
    system = ComputationSystem(rows_per_batch, num_servers, q, num_outputs, server_storage, num_partitions)
    system.storage_equal_per_server()
    system.storage_eval()
    print(repr(system))
    return
    '''

    num_runs = 100
    total_load = 0
    min_load = math.inf
    max_load = 0
    results = dict()
    for i in range(num_runs):
        system = ComputationSystem(rows_per_batch, num_servers, q, num_outputs, server_storage, num_partitions)
        #system.storage_equal_per_batch()
        #system.storage_random()
        #system.assign_batches()
        #X, B = assignmentFromBatches(system.batches, num_servers, num_partitions, int(server_storage*q))
        system.storage_equal_per_server()
        #system.storage_experiment()
        #print(system)        
        #print(repr(system))
        #system.storage_eval()
        #system.storage_random()
        #system.verify()
        #system.map(subset = (0, 1, 3, 4, 5, 8))
        #system.map(subset = (0, 1, 2, 3, 4, 5))
        selected = system.map()
        #objectiveFunction(X, B, num_source_rows, num_servers, q, Q=selected)        
        load, psi = system.shuffle()

        if load not in results:
            results[load] = 1
        else:
            results[load] = results[load] + 1
            
        print(load)
        total_load = total_load + load
        min_load = min(min_load, load)
        max_load = max(max_load, load)

    average_load = total_load / num_runs

    num_degraded = 0
    for load, count in results.items():
        if load > min_load:
            num_degraded = num_degraded + count

    for k, v in psi.items():
        print('k:', k, 'count:', v.count())

    print('-------------------')
    print('Rows per batch:', rows_per_batch)
    print('Servers:', num_servers)
    print('Wait for:', q)
    print('Number of outputs:', num_outputs)
    print('Server storage:', server_storage)
    print('Batches:', num_batches)
    print('|T|:', server_storage * q)
    print('Code rate:', q / num_servers)
    print('Source rows:', num_source_rows)
    print('Coded rows:', num_coded_rows)
    print('Partitions:', num_partitions)    
    print('Num runs:', num_runs)
    print('Min:', min_load)
    print('Max:', max_load)
    print('Average:', average_load)
    print('% degraded:', num_degraded / num_runs)

class Matrix(object):
    def __init__(self, list_of_list):
        self.rows = list_of_list

    def __str__(self):
        s = ''
        for row in self.rows:
            s = s + str(row) + '\n'

        return s

def assignmentFromBatches(batches, num_servers, num_partitions, label_size):
    num_batches = len(batches)
    
    # Initialize assignment matrix
    X = np.zeros([num_batches, num_partitions])
    assignment = [set() for x in range(num_servers)]

    row = 0
    for batch in batches:
        for symbol in batch.symbols.symbols:
            col = symbol.partition
            X[row][col] = X[row][col] + 1

        for index in batch.label:
            assignment[index].add(row)
            
        row = row + 1

    return X, assignment
            

# Assign symbols randomly
def assignmentRandom(symbols, num_batches, num_partitions, num_servers, rows_per_batch):
    # Initialize assignment matrix
    X = np.zeros([num_batches, num_partitions])
    assignment = [set() for x in range(num_servers)]    
    symbolsCopy = set(symbols)

    labels =  it.combinations(range(num_servers), 2)
    for row in range(num_batches):        
        symbol_sample = random.sample(symbolsCopy, rows_per_batch)
        while len(symbol_sample) > 0:
            symbol = symbol_sample.pop()
            symbolsCopy.remove(symbol)
            col = symbol[0]
            X[row][col] = X[row][col] + 1

        label = next(labels)
        [assignment[x].add(row) for x in label]

    return X, assignment

# Assign symbols into batches greedily
def assignmentGreedy(symbols, num_batches, num_partitions, num_servers, num_source_rows, q, rows_per_batch, server_storage):
    # First label the batches
    assignment = [set() for x in range(num_servers)]    
    labels =  it.combinations(range(num_servers), int(len(symbols) / num_batches))
    row = 0
    for label in labels:
        [assignment[x].add(row) for x in label]
        row = row + 1

    # Separate symbols by partition
    symbols_per_partition = len(symbols) / num_partitions    
    symbols_separated = [symbols_per_partition for x in range(num_partitions)]

    # Assign symbols to batches one by one, always picking the choice
    # the minimizes the objective function,
    X = np.zeros([num_batches, num_partitions])
    for row in range(num_batches):
        for i in range(rows_per_batch):
            best_col = 0
            score = math.inf

            for col in range(num_partitions):
                if symbols_separated[col] == 0:
                    continue

                X[row][col] = X[row][col] + 1
                average_score, worst_score = objectiveFunction(X,
                                                               assignment,
                                                               num_source_rows,
                                                               num_servers,
                                                               q,
                                                               server_storage,
                                                               f=remainingUnicasts)
                X[row][col] = X[row][col] - 1

                # All else being equal, favor partitions with more
                # symbols left.
                average_score = average_score - symbols_separated[col] / symbols_per_partition
                worst_score = worst_score - symbols_separated[col] / symbols_per_partition

                # Optimize over either the average or worst score
                if average_score < score:
                    best_col = col
                    score = average_score
                    
                '''
                if worst_score < score:
                    best_col = col
                    score = worst_score
                '''
            X[row][best_col] = X[row][best_col] + 1
            symbols_separated[best_col] = symbols_separated[best_col] - 1

    return X, assignment
            
    
def assignmentGreedyOld(symbols, num_batches, num_partitions, rows_per_batch):
    # Initialize assignment matrix
    X = np.zeros([num_batches, num_partitions])
    assignment = [set() for x in range(num_servers)]    

    # Separate symbols by partition
    symbols_per_partition = len(symbols) / num_partitions    
    symbols_separated = [symbols_per_partition for x in range(num_partitions)]

    col_sums = [0 for x in range(num_partitions)]
    for row in range(num_batches):
        for i in range(rows_per_batch):
            best_col = 0
            choice = None
            score = math.inf
            for col in range(num_partitions):
                if symbols_separated[col] == 0:
                    continue
            
                tentative_score = math.pow(col_sums[col] + 1, 2) + symbols_separated[col] / symbols_per_partition
                if tentative_score < score:
                    score = tentative_score
                    best_col = col

            X[row][best_col] = X[row][best_col] + 1
            #assignment[row]
            col_sums[best_col] = col_sums[best_col] + 1
            symbols_separated[best_col] = symbols_separated[best_col] - 1

    return X

# Evaluate an assignment
def assignmentEvaluate(X, num_source_rows):
    num_batches = len(X)
    num_partitions = len(X[0])
    
    # TODO: Generalize
    subsets = it.combinations(range(num_batches), 8)
    num_subsets = nchoosek(num_batches, 8)
    total_score = 0
    for c in subsets:
        s = [-num_source_rows/num_partitions for x in range(num_partitions)]
        for row in [X[x] for x in c]:
            for col in range(num_partitions):
                s[col] = s[col] + row[col]

        score = 0
        for col in range(num_partitions):
            if s[col] > 0:
                score = score + s[col]

        total_score = total_score + score

    average_score = total_score / num_subsets
    #print('Average:', average_score, 'Total:', total_score, 'Subsets:', num_subsets)
    return average_score

'''
def objectiveFunction(X):
    num_batches = len(X)
    num_partitions = len(X[0])
    
    # TODO: Generalize
    subsets = it.combinations(range(num_batches), 8)
    num_subsets = nchoosek(num_batches, 8)
    total_score = 0
    for c in subsets:
        s = [0 for x in range(num_partitions)]
        for row in [X[x] for x in c]:
            for col in range(num_partitions):
                s[col] = s[col] + row[col]

        score = 0
        for col in range(num_partitions):
            if s[col] > 0:
                score = score + math.pow(s[col], 2)

        total_score = total_score + score

    average_score = total_score / num_subsets
    #print('Average:', average_score, 'Total:', total_score, 'Subsets:', num_subsets)
    return average_score
'''

## Various performance measures
# Count the number of remaining unicasts
def remainingUnicasts(count_vector):
    unicasts = 0
    for e in count_vector[0]:
        if e < 0:
            unicasts = unicasts - e

    return unicasts

# Count the number of lost useful symbols in multicast rounds
def lostMulticasts(count_vector):
    lost_multicasts = 0
    for e in count_vector[0]:
        if e > 0:
            lost_multicasts = lost_multicasts + e

    return lost_multicasts

# Take the inner product with itself
def innerProduct(count_vector):
    return np.dot(count_vector, count_vector.T).sum()

def norm(count_vector):
    return math.sqrt(np.dot(count_vector, count_vector.T).sum())

def absoluteSum(count_vector):
    return abs(count_vector).sum()
    
def objectiveFunction(X, assignment, num_source_rows, num_servers, q, server_storage, Q=None, f=remainingUnicasts):
    num_batches = len(X)
    num_partitions = len(X[0])

    # Count the total number of extra unicasts
    total_score = 0
    worst_score = 0
    #total_unicasts = 0
    #worst_unicasts = 0
    #total_lost_multicasts = 0

    # If a specific Q was given evaluate only that one.  Otherwise
    # evaluate all possible Q.
    if Q == None:
        subsets = it.combinations(range(num_servers), q)
        num_subsets = nchoosek(num_servers, q)
    else:
        subsets = [Q]
        num_subsets = 1

    for Q in subsets:
        set_score = 0
        
        # Count over all server perspectives
        for k in Q:
            # Create the set of all symbols stored at k or sent to k
            # via multicast.            
            rows = set()
            [rows.add(row) for row in assignment[k]]

            # TODO: s_q + 1?
            sq = s_q(q, num_servers, server_storage, num_source_rows)
            for j in range(sq+1, int(server_storage*q) + 1):
                nu = set()
                for subset in it.combinations([x for x in Q if x != k], j):
                    rows = rows | set.intersection(*[assignment[x] for x in subset])

            selector_vector = np.zeros([1, num_batches])
            for row in rows:
                selector_vector[0][row] = 1

            count_vector = np.dot(selector_vector, X) - num_source_rows / num_partitions
            #score = remainingUnicasts(count_vector)
            #score = lostMulticasts(count_vector)
            #score = innerProduct(count_vector)
            score = f(count_vector)
            
            total_score = total_score + score
            set_score = set_score + score

            #print('Q:', Q, 'Selected:', selector_vector, 'Score:', score)
            
        if set_score > worst_score:
            worst_score = set_score
            
    average_score = total_score / num_subsets
    print('Average:', average_score, 'Worst:', worst_score)
    return average_score, worst_score

def objectiveValue(X, num_servers, q):
    num_batches = len(X)
    num_partitions = len(X[0])
    subsets = it.combinations(range(num_servers), q)
    num_subsets = nchoosek(num_servers, q)    
    total = 0
    for Q in subsets:
        for k in Q:
            A = np.zeros([1, num_batches])
            for i in c:
                A[0][i] = 1

        score = np.dot(A, X)
        score = np.power(score, 2)
        total_score = total_score + score
    return total_score / num_subsets

def ip_eval():
    rows_per_batch = 2
    num_servers = 6 # K in Li2016
    q = 4
    num_outputs = q # N in Li2016
    server_storage = 1/2 # \mu in Li2016    
    
    num_batches = int(scipy.misc.comb(num_servers, server_storage*q))
    num_coded_rows = rows_per_batch * num_batches
    num_source_rows = q / num_servers * num_coded_rows
    assert num_source_rows % 1 == 0
    num_source_rows = int(num_source_rows)

    # num_partitions = num_source_rows*server_storage
    num_partitions = 10
    assert num_partitions % 1 == 0    
    num_partitions = int(num_partitions)

    # Create the symbol set
    symbols = set()
    symbols_per_partition = int(num_coded_rows / num_partitions)
    for partition in range(num_partitions):
        for index in range(symbols_per_partition):
            symbols.add((partition, index))

    total_score = 0
    num_runs = 1
    for i in range(num_runs):
        #X, assignment = assignmentRandom(symbols, num_batches, num_partitions, num_servers, rows_per_batch)
        X, assignment = assignmentGreedy(symbols,
                                         num_batches,
                                         num_partitions,
                                         num_servers,
                                         num_source_rows,
                                         q,
                                         rows_per_batch,
                                         server_storage)

        average_score, worst_score = objectiveFunction(X, assignment, num_source_rows, num_servers, q, server_storage)
        total_score = total_score + average_score

    average_score = total_score / num_runs
    print('Average score:', average_score, 'Worst score:', worst_score)
    print('Winning assignment:')
    print(X)    
        

        
def load_eval():
    rows_per_batch = 2
    num_servers = 6 # K in Li2016
    q = 4
    num_outputs = q # N in Li2016
    server_storage = 1/2 # \mu in Li2016    
    
    num_batches = int(scipy.misc.comb(num_servers, server_storage*q))
    num_coded_rows = rows_per_batch * num_batches
    num_source_rows = q / num_servers * num_coded_rows
    assert num_source_rows % 1 == 0
    num_source_rows = int(num_source_rows)

    num_partitions = 5
    assert num_partitions % 1 == 0    
    num_partitions = int(num_partitions)

    sq = s_q(q, num_servers, server_storage, num_source_rows)
    multicast_batches = 0
    #for j in range(int(max(1, server_storage*q - (num_servers - q))), int(server_storage * q)):
    for j in range(sq+1, int(server_storage * q) + 1):
        multicast_batches = multicast_batches + nchoosek(q, j) * nchoosek(num_servers - q,
                                                                          int(server_storage*q) - j)
    print('mb:', multicast_batches, 'sq:', sq)
    multicast_batches = int(multicast_batches)

    # Count the unpartitioned load
    ul = unpartitioned_load(q, num_servers, server_storage, num_source_rows)    

    system = ComputationSystem(rows_per_batch, num_servers, q, num_outputs, server_storage, num_partitions)
    system.storage_random()    
    # system.storage_equal_per_batch()
    symbols = {symbol for symbol in system.symbols if symbol.output == 0}

    num_runs = 100000
    total_load = 0
    min_load = math.inf
    max_load = 0
    max_unicasts = 0
    prob = dict()
    for i in range(num_runs):
        #batches = random.sample(system.batches, multicast_batches)        
        #psi = set.union(*[batch.symbols.symbols for batch in batches])
        psi = random.sample(symbols, multicast_batches * rows_per_batch)
        #print('psi:', len(psi))
        partition_count = [-num_source_rows/num_partitions for x in range(num_partitions)]
        #print('partition_count:', partition_count)
        for partition in range(num_partitions):
            partition_count[partition] = partition_count[partition] + len({symbol for symbol in psi
                                                                       if symbol.partition == partition})
        #print('partition_count:', partition_count)

        # Count the number of additional unicasts caused by the partitioning
        unicasts = 0
        for e in partition_count:
            if e > 0:
                unicasts = unicasts + e

        #print('Added unicasts:', unicasts)
        #print('Source rows:', num_source_rows)
        if unicasts not in prob:
            prob[unicasts] = 1
        else:
            prob[unicasts] = prob[unicasts] + 1

        load = ul + unicasts / num_source_rows / num_outputs
        # print('Unpartitioned load', ul, 'Load for', num_partitions, 'partitions:', load)

        total_load = total_load + load
        min_load = min(min_load, load)
        max_load = max(max_load, load)
        max_unicasts = max(max_unicasts, unicasts)

    # Print report
    print('Average load:', total_load / num_runs, 'Min:', min_load, 'Max:', max_load, '(', max_unicasts, ')')
    print('Partitions:', num_partitions)
    for k, v in prob.items():
        print(k, ':', v / num_runs)

# Calculate the symbols received per multicast round. Defined in Li2016.    
def symbols_received_per_round(q, num_servers, server_storage, num_source_rows, multicast_index):
    code_rate = q / num_servers
    num_batches = nchoosek(num_servers, int(server_storage*q))
    rows_per_batch = num_source_rows / code_rate / num_batches
    return nchoosek(q - 1, multicast_index) * nchoosek(num_servers-q, int(server_storage*q)-multicast_index) * rows_per_batch;

# Calculate the minimum multicast index for which we need to
# multicast, s_q. Defined in Li2016.
def s_q(q, num_servers, server_storage, num_source_rows):
    needed_symbols = (1 - server_storage) * num_source_rows
    for s in range(int(server_storage * q), 0, -1):
        needed_symbols = needed_symbols - symbols_received_per_round(q,
                                                                     num_servers,
                                                                     server_storage,
                                                                     num_source_rows,
                                                                     s)
        if needed_symbols < 0:
            return s

    return 1

# Calculate B_j as defined in Li2016
def B_j(q, j, num_servers, server_storage, rows_per_batch, num_source_rows):    
    return nchoosek(q-1, j) * nchoosek(num_servers-q, int(server_storage*q)-j) * rows_per_batch / num_source_rows

# Calculate the total communication load per output vector  for an
# unpartitioned storage design as defined in Li2016.
def unpartitioned_load(q, num_servers, server_storage, num_source_rows):
    code_rate = q / num_servers
    num_batches = nchoosek(num_servers, int(server_storage*q))
    rows_per_batch = num_source_rows / code_rate / num_batches    
    sq = s_q(q, num_servers, server_storage, num_source_rows)

    L_1 = 1 - server_storage
    for j in range(sq, int(server_storage*q) + 1):
        Bj = B_j(q, j, num_servers, server_storage, rows_per_batch, num_source_rows)
        L_1 = L_1 + Bj / j
        L_1 = L_1 - Bj

    L_2 = 0
    for j in range(sq-1, int(server_storage*q) + 1):
        L_2 = L_2 + B_j(q, j, num_servers, server_storage, rows_per_batch, num_source_rows)
    
    print('L_1:', L_1, 'L_2:', L_2, 'Load per output vector.')
    return min(L_1, L_2)
    
    
