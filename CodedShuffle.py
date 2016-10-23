import numpy as np
import scipy as sp
import scipy.misc
import itertools as it
import copy

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
        return 'Symbol: {partition: ' + str(self.partition) + ' index: ' + str(self.index) + ' output: ' + str(self.output) + ' }'

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
            s = s + str(symbol)

        s = s + ' }'
        return s

    def weight(self):
        return len(self.symbols)

    # Add a symbol
    def add(self, symbol):
        assert type(symbol) is Symbol
        self.symbols.add(symbol)        

    # Peel away symbols from a collection
    def peel(self, symbol_collection):
        assert type(symbol_collection) is SymbolCollection

        # Make a copy of the local symbol set
        symbols = self.symbols.copy()

        # Discard all symbols from symbol_collection
        for symbol in symbol_collection.symbols:
            symbols.discard(symbol)

        #print('Peeled', len(self.symbols) - len(symbols), 'symbols.')

        self.symbols = symbols
        
        '''
        for symbol in symbol_collection.symbols:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
        '''
# Class representing a batch containing several symbols and labeled by
# a set of servers
class Batch(object):
    def __init__(self, label):
        assert type(label) is tuple
        
        self.symbols = SymbolCollection()
        self.label = label

    # Add a symbol to batch
    def add(self, symbol):
        assert type(symbol) is Symbol
        
        # Add symbol to list
        self.symbols.add(symbol);

# Class representing a server
class Server(object):
    def __init__(self, index, k, n, num_partitions):
        self.index = index
        self.needed = SymbolCollection()
        self.symbols = SymbolCollection()
        self.segment = SymbolCollection()
        self.received = list()
        self.outputs = list()
        self.k = k
        self.n = n
        self.num_partitions = num_partitions

    def __str__(self):
        s = '\tServer '  + str(self.index) + ': { '
        s = s + 'Symbols: ' + str(self.symbols)
        s = s + 'Needed: ' + str(self.needed)

        return s

    # Add symbols from a collection
    def add(self, symbol_collection):
        assert type(symbol_collection) is SymbolCollection

        for symbol in symbol_collection.symbols:
            self.symbols.add(symbol)

        # Discard any partitions for which the server hold enough
        # symbols.
        for partition in range(self.num_partitions):
            partition_symbols = {symbol for symbol in self.needed.symbols if symbol.partition == partition}
            if len(partition_symbols) <= (self.n - self.k):
                self.needed.symbols.discard(partition_symbols)
                
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
            print('Server', self.index, 'got a new symbol:', collection)
            self.add(collection)
            return
                            
        # Otherwise add it to the list of symbols to decode later
        print('Server', self.index, 'couldn\'t decode.')
        self.received.append(collection)
                
# Class representing a computation system
class ComputationSystem(object):
    def __init__(self, num_source_rows, num_servers, q, num_outputs, server_storage, num_partitions):
        assert type(num_source_rows) is int
        assert type(num_servers) is int
        assert type(q) is int
        assert type(num_outputs) is int
        assert type(server_storage) is float or server_storage == 1
        assert type(num_partitions) is int
        
        self.num_source_rows = num_source_rows
        self.num_servers = num_servers
        self.q = q
        self.num_outputs = num_outputs
        self.server_storage = server_storage
        self.num_partitions = num_partitions

        assert num_outputs / q % 1 == 0, 'num_outputs must be divisible by the number of servers.'
        assert server_storage * q % 1 == 0, 'server_storage * q must be integer.'

        assert num_servers / q * num_source_rows % 1 == 0, 'There must be an integer number of coded rows.'
        num_coded_rows = int(num_servers/q*num_source_rows)
        self.num_coded_rows = num_coded_rows
        
        num_batches = scipy.misc.comb(num_servers, server_storage*q)
        assert num_batches % 1 == 0, 'There must be an integer number of batches.'
        num_batches = int(num_batches)
        self.num_batches = num_batches

        if num_coded_rows / num_batches % 1 != 0:
            print('Rounding up the number of coded rows to make it a multiple of the batch size.')
            num_coded_rows = num_coded_rows + num_batches - num_coded_rows % num_batches
            
        assert num_coded_rows / num_batches % 1 == 0, 'num_coded_rows=' + str(num_coded_rows) + ' must be divisible by num_batches=' + str(num_batches) + '.'
        rows_per_batch = int(num_coded_rows / num_batches)
        self.rows_per_batch = rows_per_batch

        assert num_source_rows / num_partitions % 1 == 0, 'There must be an integer number of rows per partition.'        
        rows_per_partition = int(num_source_rows / num_partitions)
        self.rows_per_partition = rows_per_partition

        # Build a list of coded symbols
        symbols = list()
        for i in range(num_coded_rows):
            partition = i % num_partitions
            index = int(i / num_partitions)
            symbol = Symbol(partition, index, None)
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
            server = Server(server_index, num_coded_rows / num_partitions, rows_per_partition, num_partitions)
            servers.append(server)
        
        # Store symbols in batches
        symbols = iter(symbols)
        batches = list()
        batch_labels = it.combinations(range(num_servers), int(server_storage*q))   
        for batch_index in range(num_batches):
            # Label the batch by a unique subset of servers
            label = next(batch_labels)
            batch = Batch(label)
        
            for symbol_index in range(rows_per_batch):
                symbol = next(symbols)
                batch.add(symbol)

            batches.append(batch)

        # Store batches at servers
        for batch in batches:
            for server_index in batch.label:
                server = servers[server_index]
                server.add(batch.symbols)

    def __str__(self):
        s = 'Computation System:\n'
        for server in self.servers:
            s = s + str(server) + '\n'

        return s

    # Verify that the storage design is feasible in the sense that any
    # q servers hold enough rows to decode all partitions.    
    def verify(self):
        print('Verifying storage scheme...')
        valid = True                    
        
        # Construct an iterator of all subsets of q servers
        server_subsets = it.combinations(range(self.num_servers), self.q)

        # Iterate over the possible cardinality q subsets
        for subset in server_subsets:

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

            # Count the number of symbols per partition
            for partition in range(self.num_partitions):
                partition_symbols = [symbol for symbol in symbols if symbol.partition == partition]
                unique_symbols_partition = len(partition_symbols)
                if unique_symbols_partition < (self.num_source_rows / self.num_partitions):
                    print('ERROR: Storage holds', unique_symbols_partition,
                          'for partition', partition, 'but needs', self.num_source_rows / self.num_partitions)
                    valid = False
                    break

            # TODO: Should we always break after 1 subset
            break
            
        print('Storage is valid:', valid)

    # Perform map phase
    def map(self):
        print('Performing map phase...')
        
        # Select q random servers
        # TODO: Select a random set instead of always the first one
        server_subsets = it.combinations(range(self.num_servers), self.q)
        subset = next(server_subsets)

        outputs_per_server = int(self.num_outputs / self.q)

        # Make a list of the finished servers
        self.finished_servers = [self.servers[index] for index in subset]

        # Assign outputs to the servers
        # TODO: Find a more pythonic way of doing this
        output_index = 0
        for server in self.finished_servers:
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
            server.needed.symbols = {symbol for symbol in self.symbols if symbol not in server.symbols.symbols and symbol.output in server.outputs}
            
            # Remove the partitions where the server hold enough
            # symbols already
            for partition in range(self.num_partitions):
                partition_symbols = {symbol for symbol in server.needed.symbols if symbol.partition == partition}
                if len(partition_symbols) <= (self.num_coded_rows - self.num_source_rows) / self.num_partitions:
                    server.needed.symbols.discard(partition_symbols)

    # Perform shuffle phase
    def shuffle(self):
        print('Performing shuffle phase...')
        
        # Count the transmissions
        transmissions = 0

        multicasting_index = int(self.server_storage * self.q) # j in Li2016
        multicasting_subsets = it.combinations(range(self.q), multicasting_index+1) # S in li 0216
        for subset in multicasting_subsets:
            #print('subset:', subset)
            
            for k in range(len(subset)):
                #print('k=', k)
                #print('servers_without_k=', subset[0:k] + subset[k+1:])
                
                server_k = self.finished_servers[subset[k]]
                servers_without_k = [self.finished_servers[index] for index in subset[0:k] + subset[k+1:]]

                # The symbols needed by k
                nu = server_k.needed.symbols.copy()

                # Intersected by the set of symbols known by the others
                for server in servers_without_k:
                    nu = nu & server.symbols.symbols

                #print('nu=', nu)

                # Split nu evenly over servers_without_k
                server_cycle = it.cycle(servers_without_k)
                while len(nu) > 0:
                    server = next(server_cycle)
                    server.segment.add(nu.pop())

            print('---- Multicasting Round ----')                
            for server in [self.finished_servers[index] for index in subset]:

                # Transmit as long as there are symbols left in this
                # server's segment
                while len(server.segment.symbols) > 0:
                
                    # Select one symbol for every receiving server
                    collection = SymbolCollection()
                    for receiving_server in [self.finished_servers[index] for index in subset if self.finished_servers[index].index != server.index]:
                        symbol = {symbol for symbol in server.segment.symbols if symbol in receiving_server.needed.symbols}.pop()
                        #print('symbol=', symbol)
                        collection.add(symbol)
                        server.segment.symbols.remove(symbol)

                    # Multicast collection
                    self.multicast(server.index, collection)
                    transmissions = transmissions + 1

                # Reset segment
                server.segment = SymbolCollection()

        print('Shuffle phase ended after', transmissions, 'transmission.')
            

    # Multicast a symbol from one server to the others
    def multicast(self, source, symbol_collection):
        print('Multicasting', symbol_collection, 'from server', source)
        receveing_servers = [server for server in self.finished_servers if server.index != source]
        for server in receveing_servers:
            server.receive(symbol_collection)
            
def main():
    num_source_rows = 360 # m in Li2016
    num_servers = 18 # K in Li2016
    q = 12
    num_outputs = 180 # N in Li2016
    server_storage = 1/3 # \mu in Li2016
    num_partitions = 1
    system = ComputationSystem(num_source_rows, num_servers, q, num_outputs, server_storage, num_partitions)
    system.verify()
    system.map()
    system.shuffle()
        
