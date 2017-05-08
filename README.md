# Coded Shuffling Solvers and Simulator
This project is a Python toolbox for research on coding theory for distributed computing. It was first developed as part of the research for [1][1], but is meant to provide general useful tools for the area. Specifically it implements

## Modules
* model: System model description and analytic expressions for computing computational delay and load of the unpartitioned scheme.
* complexity: Functions used to compute the complexity of matrix multiplication and decoding operations.
* numtools: Numerical toolbox. Currently only provides a means of numerically inverting a function.
* simulation: High-level code for running simulations. Provides code for finding and evaluating assignments for several parameters in parallel.
* examples: Example usage. This is the code used to generate the plots in our paper.

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
If you have Make installed you can run the tests by typing ```make test``` in the project root folder. Run the example code by entering ```python3 examples.py```.

# Datasets
The data from our simulations is available [here](https://www.dropbox.com/sh/4w0rv9r04eynu2f/AAC6RhLETeokEkxHThQgquyQa?dl=0). Download and extract the folder to the same directory as the source code to use it when running the example code.

# References
[1]: https://arxiv.org/abs/1701.06631
