# Coded Shuffling Solvers and Simulator
This repo contains the code relating to the article written by Albin Severinson, Alexandre Graell i Amat, and Eirik Rosnes on block-diagonal coding for distributed computation with straggling servers. We present the code in the hope that it can be useful for someone else wanting to pursue this line of research.

This works builds on the concepts introduced in the paper "A Unified Coding Framework for Distributed Computing with Straggling Servers", by Songze Li, Mohammad Ali Maddah-Ali, and A. Salman Avestimehr. The paper is available on arXiv https://arxiv.org/abs/1609.01690, and we recommend reading the paper to understand how the scheme works.

## Files
* model.py: Contains the system model description and integer programming formulation.
* solvers.py: Optimization solvers for the storage design problem and functions relating to performance evaluation.
* examples.py: Example usage. This is the code used to run the simulations and generate the plots in our paper.

## Setup
The code is written for Python 3, and you need to following modules:
* scipy
* numpy
* pandas
* matplotlib
* pyplot

If you have pip3 (a Python 3 package manager) installed, you can install the modules by running ```pip3 install <modulename>``` in a terminal.

## Running
Run the example code by opening a terminal in the directory containing the code and entering ```python3 examples.py```.

## Datasets
The data from our simulations is available [here](https://www.dropbox.com/sh/4w0rv9r04eynu2f/AAC6RhLETeokEkxHThQgquyQa?dl=0). Download and extract the folder to the same directory as the source code to use it when running the example code.