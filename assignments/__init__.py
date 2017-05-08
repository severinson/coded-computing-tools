'''An assignment stores which partitions are stored in which batch,
and at what servers. Ths init file contains the assignment abstract
base class.

'''

from abc import ABC, abstractmethod

class AssignmentError(Exception):
    '''Base class for exceptions thrown by this package.'''

class Assignment(ABC):
    '''Abstract superclass representing an assignment of encoded rows into
    batches.

    '''
    @abstractmethod
    def increment(self, rows, cols, values):
        pass

    @abstractmethod
    def decrement(self, rows, cols, values):
        pass

    @abstractmethod
    def batch_union(self, batch_indices):
        pass

    @abstractmethod
    def save(self, directory='./saved_assignments/'):
        pass

    @abstractmethod
    def load(self, par, directory='./saved_assignments/'):
        pass
