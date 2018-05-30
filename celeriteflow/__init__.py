# -*- coding: utf-8 -*-

from __future__ import division, print_function
__version__ = "0.0.1"

try:
    __CELERITEFLOW_SETUP__
except NameError:
    __CELERITEFLOW_SETUP__ = False

if not __CELERITEFLOW_SETUP__:
    __all__ = ["CeleriteFlowMatrix", "to_dense", "get_matrices", "factor", "solve"]

    from .celeriteflow import CeleriteFlowMatrix
    from .ops import to_dense, get_matrices, factor, solve
