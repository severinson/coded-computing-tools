# Coded Computing Tools
This project is a toolbox for research on coding theoretic applications to computing. It was first developed as part of the research for [1][1], but is meant to provide a general suite of tools. It's written in Python and includes the modules and packages described below.

## Modules
* model: System model description and analytic expressions for computing computational delay and load of the unpartitioned scheme.
* complexity: Functions used to compute the complexity of matrix multiplication and decoding operations.
* numtools: Numerical tools used by the other modules. Such as code for numerically inverting a function.
* simulation: High-level code for running simulations. Provides code for finding and evaluating assignments for several parameters in parallel.
* examples: Example usage. This is the code used to generate the plots in our paper.
* lt: Experimental support for fountain code analysis.

## Packages
* assignments: This package provides implementations of assignment matrices.
* solvers: Optimization solvers used to find assignment matrices.
* evaluation: Contains code for numerically evaluating the performance of an assignment matrix.
* tests: All tests reside in this package.

# Setup
The code is written for Python 3, and you need to following modules:
* scipy
* numpy
* pandas
* matplotlib
* pyplot

If you have pip3 (a Python 3 package manager) installed, you can install the modules by running ```pip3 install <modulename>``` in a terminal.

# Running
If you have Make installed you can run the tests by typing ```make test``` in the project root folder. The results of our simulations is included in the results.tar.gz archive. Extract the contents to the current directory and show the plots by entering ```python3 examples.py```.

[1]: https://arxiv.org/abs/1701.06631
