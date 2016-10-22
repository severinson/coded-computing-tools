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
        assert type(output) is int
            
        self.partition = partition
        self.index = index
        self.output = output

    def __str__(self):
        return 'Symbol: { partition: ' + str(self.partition) + ' index: ' + str(self.index) + ' }'

# Class representing a single output symbol
class OutputSymbol(object):
    def __init__(self, symbol, output):
        assert type(symbol) is Symbol
        assert type(output) is int        
        
        self.symbol = symbol        
        self.output = output

    def __str__(self):
        s = 'OutputSymbol:'
        s = s + '{ partition: ' + str(self.symbol.partition) + ' index: ' + str(self.symbol.index) + ' output: ' + str(self.output) + ' }'

        return s

# Class representing a multicasting symbol
class MulticastingSymbol(object):
    def __init__(self):
        self.partitions = dict()
        self.weight = 0

        self.output_symbols = list()

    def __str__(self):
        s = 'MulticastingSymbol:'
        s = s + ' partitions=['        
        for partition in self.partitions:
            s = s + str(partition) + ': '
            s = s + 'indices=['
            for index in self.partitions[partition]:
                s = s + str(index) + ': '
                s = s + 'outputs=['        
                for output in self.partitions[partition][index]:
                    s = s + str(output) + ', '
                s = s + '], '
            s = s + '], '
        s = s + ']'

        return s

    # Add a symbol
    def add(self, output_symbol):
        assert type(output_symbol) is OutputSymbol

        partition = output_symbol.symbol.partition
        index = output_symbol.symbol.index
        output = output_symbol.output

        if partition not in self.partitions:
            self.partitions[partition] = dict()

        if index not in self.partitions[partition]:
            self.partitions[partition][index] = dict()

        self.partitions[partition][index][output] = 1

        self.output_symbols.append(output_symbol)

        # Increment weight
        self.weight = self.weight + 1

    # Peel away symbols in partition
    def peel(self, partitions):
        assert type(partitions) is dict

        # Make a copy of the dict
        self.partitions = copy.deepcopy(self.partitions)
        
        for partition in partitions:
            if partition not in self.partitions:
                continue

            for index in partitions[partition]:
                if index not in self.partitions[partition]:
                    continue

                for output in partitions[partition][index]:
                    if output not in self.partitions[partition][index]:
                        continue

                    # Peel away symbol
                    del self.partitions[partition][index][output]

                    # Decrement weight
                    self.weight = self.weight - 1

                    #print('Peeled partition=', partition, ' index=', index, ' output=', output)

                if len(self.partitions[partition][index]) == 0:
                    del self.partitions[partition][index]

            if len(self.partitions[partition]) == 0:
                del self.partitions[partition]

# Class representing a batch containing several symbols
class Batch(object):
    def __init__(self, label):
        assert type(label) is tuple
        
        self.symbols = list()
        self.partitions = dict()
        self.label = label

    def __str__(self):
        s = 'Batch: partitions=['
        for partition in self.partitions:
            s = s + str(partition) + ': indices=['
            for index in self.partitions[partition]:
                s = s + str(index) + ', '
            s = s + '], '
        s = s + ']'
        return s

    # Add a symbol to batch
    def store(self, symbol):
        assert type(symbol) is Symbol
        
        # Add symbol to list
        self.symbols.append(symbol);

        # Update indices
        if symbol.partition not in self.partitions:
            self.partitions[symbol.partition] = dict()

        if symbol.index not in self.partitions[symbol.partition]:
            self.partitions[symbol.partition][symbol.index] = dict()

        #self.partitions[symbol.partition][symbol.index] = self.partitions[symbol.partition][symbol.index] + 1
    

# Class representing a server containing several batches
class Server(object):
    def __init__(self, index):
        self.index = index
        self.batches = list()
        self.partitions = dict()
        self.outputs = list()
        self.needed = dict()
        self.segment = dict()
        self.multicast = dict()
        self.received = list()
        self.remaining = dict()
        self.k = None
        self.n = None

    def __str__(self):
        s = 'Server: partitions=['
        for partition in self.partitions:
            s = s + str(partition) + ': indices=['
            for index in self.partitions[partition]:
                s = s + str(index) + ', '
            s = s + '], '

        s = s + '], '
        s = s + 'outputs=['
        for output in self.outputs:
            s = s + str(output) + ', '

        s = s + '], '
        s = s + 'needed=['
        for partition in self.needed:
            s = s + str(partition) + ': ' + str(self.needed[partition]) + ', '

        s = s + '], '
        s = s + 'remaining:['
        for output in self.remaining:
            s = s + 'output=[' + str(output) + ': '
            s = s + str(self.remaining[output])
        s = s + '], '
        
        '''
        s = s + '], '        
        s = s + 'segment=['
        for output in self.segment:
            s = s + 'output=' + str(output) + ' ' + str(self.segment[output])
            s = s + ', '
        '''
        
        s = s + ']'

        return s

    # Add symbols from a dict
    def add(self, partitions):
        for partition in partitions:
            if partition not in self.partitions:
                self.partitions[partition] = dict()
                
            for index in partitions[partition]:
                if index not in self.partitions[partition]:
                    self.partitions[partition][index] = dict()

                    # TODO: Make compatible with N > q
                    self.needed[partition] = self.needed[partition] - 1
                    
                for output in partitions[partition][index]:
                    self.partitions[partition][index][output] = 1

                    # Remove the symbol from the list of needed
                    # symbols
                    if output in self.remaining:
                        if partition in self.remaining[output]:
                            if index in self.remaining[output][partition]:
                                del self.remaining[output][partition][index]

                            # Remove the partition if we hold enough
                            # symbols to decode.
                            if len(self.remaining[output][partition]) <= (self.n - self.k):
                                del self.remaining[output][partition]

                        # Remove the output if we need no partitions
                        if len(self.remaining[output]) == 0:
                            del self.remaining[output]
                
    # Receive a multicast symbol
    def receive(self, multicasting_symbol):
        assert type(multicasting_symbol) is MulticastingSymbol

        # Make a copy of the symbol
        own_symbol = copy.copy(multicasting_symbol)

        # Peel away any known symbols
        own_symbol.peel(self.partitions)

        # If all symbols were removed, return
        if own_symbol.weight == 0:
            return

        # If only 1 symbol remains, add it to the known symbols
        if own_symbol.weight == 1:
            print('Server', self.index, 'got a new symbol:', own_symbol)
            self.add(multicasting_symbol.partitions)            
            return
                            
        # Otherwise add it to the list of symbols to decode later
        print('Server', self.index, 'couldn\'t decode.')
        self.received.append(own_symbol)
        

                
# Setup the storage design and return a list of servers
class Storage(object):
    def __init__(self):
        m = 20 # Number of source rows
        self.m = m

        num_servers = 6 # Total servers
        self.num_servers = num_servers
        q = 4 # Servers to wait for
        self.q = q

        N = q # Number of input/output rows
        self.N = N        

        if (N/q) % 1 != 0:
            print('ERROR: N/q must be integer.')
            return
        
        mu = 1/2 # Server storage
        self.mu = mu
        if (mu*q) % 1 != 0:
            print('ERROR: mu*q must be integer.')
            return
    
        num_partitions = 1 # Number of partitions
        self.num_partitions = num_partitions
        
        coded_rows = num_servers/q*m;
        if coded_rows % 1 != 0:
            print('ERROR: There must be an even number of coded rows.')
            return

        coded_rows = int(coded_rows)
        self.coded_rows = coded_rows
        
        num_batches = scipy.misc.comb(num_servers, mu*q)
        rows_per_batch = coded_rows / num_batches

        # Input validation
        if coded_rows % 1 != 0:
            print('ERROR: Parameters m=' + str(m) + ', num_servers=' + str(num_servers) + ', q=' + str(q),
                  'result in an uneven number of coded rows.')
            return
        coded_rows = int(coded_rows)

        rows_per_partition = m / num_partitions
        if rows_per_partition % 1 != 0:
            print('ERROR: Parameters m=' + str(m) + ', T=' + str(num_partitions),
                  'result in an uneven number of rows per partition.')
            return
        rows_per_partition = int(rows_per_partition)
        self.rows_per_partition = rows_per_partition

        if num_batches % 1 != 0:
            print('ERROR: Uneven number of batches.')
            return
        num_batches = int(num_batches)

        if rows_per_batch % 1 != 0:
            print('ERROR: Parameters result in an uneven number of rows per batch.')
            return
        rows_per_batch = int(rows_per_batch)

        # Build a list of coded symbols
        symbols = list()
        for i in range(coded_rows):
            partition = i % num_partitions
            index = int(i / num_partitions)
            symbol = Symbol(partition, index)
            symbols.append(symbol)

        # Setup servers
        servers = list()
        for server_index in range(num_servers):
            server = Server(server_index)
            server.n = coded_rows / num_partitions
            server.k = rows_per_partition

            # Count how many symbols the server needs
            for partition_index in range(num_partitions):
                server.needed[partition_index] = rows_per_partition
            
            servers.append(server)
        
        # Store symbols in batches
        symbols = iter(symbols)
        batches = list()
        batch_labels = it.combinations(range(num_servers), int(mu*q))   
        for batch_index in range(num_batches):
            # Label the batch by a unique subset of servers and create it        
            label = next(batch_labels)
            batch = Batch(label) 
        
            for symbol_index in range(rows_per_batch):
                symbol = next(symbols)
                batch.store(symbol)

            batches.append(batch)

        # Store batches at servers
        for batch in batches:
            for server_index in batch.label:
                server = servers[server_index]
                server.add(batch.partitions)

        self.servers = servers
        
    def __str__(self):
        s = 'Storage: servers=[\n'
        for server in self.servers:
            s = s + str(server) + '\n'
        s = s + ']'
        return s
    

    # Verify that the storage is feasible in the sense that any q
    # servers hold enough rows to decode all partitions.
    def verify(self):
        print('Verifying storage scheme...')
        
        # Construct an iterator of all subsets of q servers
        server_subsets = it.combinations(range(self.num_servers), self.q)

        # Iterate over the possible cardinality q subsets
        for subset in server_subsets:
            partitions = dict()

            # Iterate over the servers in the subset
            for server_index in subset:
                server = self.servers[server_index]

                # For every server, iterate over the partitions it stores                
                for partition in server.partitions:
                    if partition not in partitions:
                        partitions[partition] = dict()

                    # For every partition, iterate over the indices
                    for index in server.partitions[partition]:
                        if index not in partitions[partition]:
                            partitions[partition][index] = 0

                        partitions[partition][index] = partitions[partition][index] + 1

            # Verify that there's enough symbols from every partition
            for partition_index in range(self.num_partitions):
                valid = True
                if partition_index not in partitions:
                    print('WARN: Storage not valid. No symbols from partition',
                          partition_index + '.')
                    valid = False
                
                num_symbols = len(partitions[partition_index])
                if num_symbols < self.rows_per_partition:
                    print('WARN: Storage not valid. Not enough symbols from partition',
                          partition_index + '. Got', num_symbols, 'but needed',
                          self.rows_per_partition + '.')
                    valid = False               
                
            print(subset, 'VALID:', valid)

    # Perform map phase
    def map(self):
        print('Performing map phase...')
        
        # Select q random servers
        # TODO: Select a random set instead of always the first one
        server_subsets = it.combinations(range(self.num_servers), self.q)
        subset = next(server_subsets)

        self.finished_servers = list()

        for server_index in subset:
            server = self.servers[server_index]

            # Assign output vectors to servers
            output_rows_per_server = int(self.N/self.q)
            for output_index in range(server_index * output_rows_per_server,
                                      (server_index + 1) * output_rows_per_server):
                server.outputs.append(output_index)

            # Add the outputs to the symbols of the server
            for partition in server.partitions:
                for index in server.partitions[partition]:
                    server.partitions[partition][index] = dict()
                    for output in range(self.N):
                        server.partitions[partition][index][output] = 1

            # Figure out which symbols this server needs
            # All outputs it needs
            for output in server.outputs:
                server.remaining[output] = dict()

                # All partitions
                for partition in range(self.num_partitions):
                    server.remaining[output][partition] = dict()

                    # If the server doesn't hold any indices from the
                    # partition, add them all.
                    if partition not in server.partitions:
                        for index in range(int(self.coded_rows / self.num_partitions)):
                            server.remaining[output][partition][index] = 1

                    # Otherwise, only add the missing ones.
                    else:
                        for index in range(int(self.coded_rows / self.num_partitions)):
                            if index in server.partitions[partition]:
                                continue
                            
                            server.remaining[output][partition][index] = 1                        

                    # Remove all partitions for which the server hold
                    # enough symbols.
                    if len(server.remaining[output][partition]) <= (self.coded_rows / self.num_partitions - self.rows_per_partition):
                        del server.remaining[output][partition]

                # Remove outputs for which the server needs no partitions.
                if len(server.remaining[output]) == 0:
                    del server.remaining[output]
            
            # TODO: Clean up this part
            # TODO: I'm not using this anyway at the moment...
            '''
            for partition_index in range(self.num_partitions):
                server.needed[partition_index] = dict()
                if partition_index not in server.partitions:
                    for symbol_index in range(int(self.coded_rows / self.num_partitions)):
                        server.needed[partition_index][symbol_index] = 1
                else:
                    for symbol_index in range(int(self.coded_rows / self.num_partitions)):
                        if symbol_index not in server.partitions[partition_index]:
                            server.needed[partition_index][symbol_index] = 1
            '''                              
            self.finished_servers.append(server)
            print(server)

    # Perform shuffle phase
    def shuffle(self):
        print('Performing shuffle phase...')
        
        # Count the transmissions
        transmissions = 0

        multicasting_index = int(self.mu * self.q) # j in Li2016
        multicasting_subsets = it.combinations(range(self.q), multicasting_index+1) # S in li 0216
        for subset in multicasting_subsets:
            print('subset:', subset)
            
            for k in range(len(subset)):
                one = self.finished_servers[subset[k]]
                subset_without_one = subset[0:k] + subset[k+1:]

                print('k:', k)                
                print('subset_without_one:', subset_without_one)                

                # Construct the set nu of symbols needed by k and
                # known exclusively in the set multicasting_subsets
                # without k
                nu = dict()

                '''
                # Reorganize the server partitions dict
                # TODO: Use this format all the time
                foo = dict()
                for server_index in subset_without_one:
                    foor[server_index] = dict()
                    
                    server = self.finished_servers[server_index]
                    for partition in server.partitions:
                        for index in server.partitions[partition]:
                            for output in server.partitions[partition][index]:
                                if output not in foo[server_index]:
                                    foo[server_index][output] = dict()

                                if partition not in foo[server_index][output]:
                                    foo[server_index][output][partition] = dict()

                                if index not in foo[server_index][output][partition]:
                                    foo[server_index][output][partition][index] = 1



                # Outputs needed by k
                nu_output_keys = set(one.remaining.keys())

                # Intersected with outputs known by the others
                for server in foo:
                    nu_output_keys = nu_output_keys & set(foo[server].keys())

                for output in nu_output_keys:
                    nu[output] = dict()

                    # Partitions needed by k
                    nu_partition_keys = set(one.remaining[output].keys())

                    # Intersected by partitions known by the others
                    for server in foo:
                        nu_partition_keys = nu_partition_keys & set(foo[server].keys())

                    
                
                # Build a list of partitions needed by k and known by
                # the others:
                nu_partition_keys = set()

                # The union of partitions needed by k
                for output in one.remaining:
                    # Set union
                    nu_partition_keys = nu_partition_keys | set(one.remaining[output].keys())

                # The intersection of partitions known by the others
                for server_index in subset_without_one:
                    server = self.finished_servers[server_index]
                    nu_partition_keys = nu_partition_keys & set(server.partitions.keys())

                # For every partition, we need to check the indices
                for partition in nu_partition_keys:
                    nu[partition] = dict()
                    nu_index_keys = set()
                    for output in one.remaining:
                        if partition not in one.remaining[output]:
                            continue

                        # Union of all indices needed by k
                        nu_index_keys = nu_index_keys | set(one.remaining[partition].keys())

                        
                        
                        for partition in one.remaining[partition]:

                    
                for output in one.remaining:
                    for partition in one.remaining[output]:
                        
                        for index in one.remaining[output][partition]:
                            
                            
                '''


                first_server = self.finished_servers[subset_without_one[0]]
                nu_partition_keys = set(first_server.partitions.keys())
                for server_index in subset_without_one[1:]:
                    server_partition_keys = set(self.finished_servers[server_index].partitions.keys())
                    nu_partition_keys = nu_partition_keys & server_partition_keys

                for partition_key in nu_partition_keys:
                    nu[partition_key] = dict()
                    nu_index_keys = set(first_server.partitions[partition_key])
                    for server_index in subset_without_one[1:]:
                        server_index_keys = set(self.finished_servers[server_index].partitions[partition_key].keys())
                        nu_index_keys = nu_index_keys & server_index_keys

                    for index_key in nu_index_keys:
                        nu[partition_key][index_key] = dict()

                for partition in nu:
                    for index in nu[partition]:
                        nu_output_keys = set(first_server.partitions[partition][index].keys())
                        for server_index in subset_without_one[1:]:
                            server_output_keys = set(self.finished_servers[server_index].partitions[partition][index].keys())
                            nu_output_keys = nu_output_keys & server_output_keys

                        for output in nu_output_keys:
                            if output not in one.outputs:
                                continue
                            
                            nu[partition][index][output] = 1

                for partition in range(self.num_partitions):
                    if partition not in nu:
                        continue
                    
                    for index in range(int(self.coded_rows / self.num_partitions)):
                        if index not in nu[partition]:
                            continue

                        if len(nu[partition][index]) == 0:
                            del nu[partition][index]

                    if len(nu[partition]) == 0:
                        del nu[partition]


                print('nu before delete:', nu)
                print('one:', one)
                
                # Remove all symbols server k doesn't need                              
                for partition in one.partitions:
                    if partition not in nu:
                        continue
                    
                    for index in one.partitions[partition]:
                        if index not in nu[partition]:
                            continue
                            
                        print('Deleting partition=', partition, 'index=', index)
                        del nu[partition][index]

                print('nu:', nu)
                # Split nu evenly over subset_without_one
                server_index_foo = 0
                for partition in nu:
                    for index in nu[partition]:
                        server_index = subset_without_one[server_index_foo % len(subset_without_one)]
                        server_index_foo = server_index_foo + 1

                        server = self.finished_servers[server_index]
                        symbol = Symbol(partition, index)

                        if k not in server.multicast:
                            server.multicast[k] = list()

                        for output in one.outputs:
                            output_symbol = OutputSymbol(symbol, output)
                            server.multicast[k].append(output_symbol)
                                        
                    
            # Multicast
            print('---- Multicasting Round: ----')
            for server in self.finished_servers:                
                server.segment = dict()
                for k in server.multicast:
                    print('k=', k)
                    
                    multicast_index = 0
                    for output_symbol in server.multicast[k]:
                        print('output_symbol:', output_symbol)

                        if multicast_index not in server.segment:
                            server.segment[multicast_index] = MulticastingSymbol()

                        server.segment[multicast_index].add(output_symbol)
                        multicast_index = multicast_index + 1
                        
                for multicast_index in server.segment:
                    self.multicast(server.index, server.segment[multicast_index])
                    transmissions = transmissions + 1

                server.multicast = dict()
                print('')

        print('Shuffle phase finished after', transmissions, 'multicasts.')

        # Count the number of remaining symbols
        
        
        print(self)

    # Multicast a symbol from one server to the others
    def multicast(self, source, multicasting_symbol):
        print('Multicasting', multicasting_symbol, 'from server', source)
        for server in self.finished_servers:
            server.receive(multicasting_symbol)

def main():
    storage = Storage()
    print(storage)

    storage.verify()
    storage.map()
    storage.shuffle()

