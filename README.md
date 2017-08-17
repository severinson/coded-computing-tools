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
With Python 3 installed, setup the project by running `setup.sh`. This creates a Python 3 virtual environment (an isolated environment for Python to store dependencies in) and automatically installs all dependencies. Activate the virtual environment by typing `source venv/bin/activate`. To avoid having to re-run the simulations, download the data used in our paper [here](https://www.dropbox.com/s/8oc40l3m9ksxah8/results.tar.gz?dl=0) and extract the contents to the project folder. Next, show the plots by running `python3 examples.py`. Type `deactivate` to deactivate the virtual environment.

# Test
Test the code by running `make test`. Note that there is a failing test for test_lt as fountain codes are not yet well supported.

[1]: https://arxiv.org/abs/1701.06631
