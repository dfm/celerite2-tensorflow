# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["CeleriteFlowMatrix"]

import tensorflow as tf
from .ops import to_dense, get_matrices, factor, solve, matmul


class CeleriteFlowMatrix(object):

    def __init__(self, a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, diag,
                 name=None):
        with tf.name_scope(name, "CeleriteFlowMatrix"):
            matrices = get_matrices(a_real, c_real, a_comp, b_comp, c_comp,
                                    d_comp, x, diag)
            self.a, self.U, self.V, self.P = matrices
            self.d, self.W = factor(self.a, self.U, self.V, self.P,
                                    name="factor")
            self.log_determinant = tf.reduce_sum(tf.log(self.d),
                                                 name="log_determinant")
            self.dense = to_dense(self.a, self.U, self.V, self.P,
                                  name="dense")

    def apply_inverse(self, y, **kwargs):
        return solve(self.U, self.P, self.d, self.W, y, **kwargs)

    def matmul(self, z, **kwargs):
        return matmul(self.a, self.U, self.V, self.P, z, **kwargs)
