# Coded Shuffling Solvers and SImulator
This repo contains the code relating to the article written by Albin Severinson, Alexandre Graell i Amat, and Eirik Rosnes on block-diagonal coding for distributed computation.

It includes three files:
* main.py: Contains the system model description and functions relating to performance measurement.
* solver.py: Optimization solvers for the storage design problem.
* simulator.py: Shuffling simulator.
* examples.py: Example usage.

This works builds on the concepts introduced in the paper "A Unified Coding Framework for Distributed Computing with Straggling Servers", by Songze Li, Mohammad Ali Maddah-Ali, and A. Salman Avestimehr. The paper is available on arXiv https://arxiv.org/abs/1609.01690, and we recommend reading the paper to understand how the scheme works.

# This is the main file for all code relating to the paper by Albin
# Severinson, Alexandre Graell i Amat, and Eirik Rosnes. If you want
# to re-create the simulations in the paper, or want to simulate the
# system for other parameters, this is where you start.

# This file contains the code relating to the integer programming
# formulation used in the paper by Albin Severinson, Alexandre Graell
# i Amat, and Eirik Rosnes. This includes the formulations itself, and
# the solvers presented in the article.

# This is a simulator of the unified coding scheme presented in
# the article "A Unified Coding Framework for Distributed Computing
# with Straggling Servers", by Songze Li, Mohammad Ali Maddah-Ali, and
# A. Salman Avestimehr.
# The paper is available on arXiv https://arxiv.org/abs/1609.01690,
# and we recommend reading the paper to understand how the scheme
# works.