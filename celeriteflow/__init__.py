# -*- coding: utf-8 -*-

from __future__ import division, print_function
__version__ = "0.0.1"

try:
    __CELERITEFLOW_SETUP__
except NameError:
    __CELERITEFLOW_SETUP__ = False

if not __CELERITEFLOW_SETUP__:
    __all__ = ["terms",
               "to_dense", "factor", "solve", "matmul",
               "GaussianProcess", "Solver", "get_matrices", ]

    from . import terms
    from .ops import to_dense, factor, solve, matmul
    from .celeriteflow import GaussianProcess, Solver, get_matrices
