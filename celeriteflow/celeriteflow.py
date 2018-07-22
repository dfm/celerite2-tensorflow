# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Solver", "get_matrices"]

import numpy as np
import tensorflow as tf
from .ops import to_dense, factor, solve, matmul


def get_matrices(a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, diag,
                 name=None):
    with tf.name_scope(name, "get_matrices"):
        a = tf.add(diag, tf.reduce_sum(a_real) + tf.reduce_sum(a_comp),
                   name="a")

        U = tf.concat((
            a_real[None, :] + tf.zeros_like(x)[:, None],
            a_comp[None, :] * tf.cos(d_comp[None, :] * x[:, None])
            + b_comp[None, :] * tf.sin(d_comp[None, :] * x[:, None]),
            a_comp[None, :] * tf.sin(d_comp[None, :] * x[:, None])
            - b_comp[None, :] * tf.cos(d_comp[None, :] * x[:, None]),
        ), axis=1, name="U")

        V = tf.concat((
            tf.zeros_like(a_real)[None, :] + tf.ones_like(x)[:, None],
            tf.cos(d_comp[None, :] * x[:, None]),
            tf.sin(d_comp[None, :] * x[:, None]),
        ), axis=1, name="V")

        dx = x[1:] - x[:-1]
        P = tf.concat((
            tf.exp(-c_real[None, :] * dx[:, None]),
            tf.exp(-c_comp[None, :] * dx[:, None]),
            tf.exp(-c_comp[None, :] * dx[:, None]),
        ), axis=1, name="P")

        return a, U, V, P


class Solver(object):

    def __init__(self, kernel, x, diag, name=None):
        self.name = name
        self.kernel = kernel
        with tf.name_scope(self.name, "Solver"):
            matrices = get_matrices(
                *(list(self.kernel.get_coefficients()) + [x, diag]))
            self.a, self.U, self.V, self.P = matrices
            self.d, self.W = factor(self.a, self.U, self.V, self.P,
                                    name="factor")
            self.log_determinant = tf.reduce_sum(tf.log(self.d),
                                                 name="log_determinant")
            self.dense = to_dense(self.a, self.U, self.V, self.P,
                                  name="dense")

    def apply_inverse(self, y, **kwargs):
        with tf.name_scope(self.name, "apply_inverse"):
            return solve(self.U, self.P, self.d, self.W, y, **kwargs)

    def matmul(self, z, **kwargs):
        with tf.name_scope(self.name, "matmul"):
            return matmul(self.a, self.U, self.V, self.P, z, **kwargs)


class GaussianProcess(object):

    def __init__(self, kernel, x, diag, name=None):
        self.solver = Solver(kernel, x, diag, name=name)
        self.constant = tf.cast(tf.size(x), x.dtype) \
            * tf.constant(np.log(2*np.pi), dtype=x.dtype)

    def log_likelihood(self, y, name=None):
        with tf.name_scope(name, "log_likelihood"):
            return -0.5 * (
                tf.squeeze(
                    tf.matmul(tf.transpose(y),
                              self.solver.apply_inverse(y)))
                + self.solver.log_determinant + self.constant)

    def predict(self, y, name=None):
        with tf.name_scope(name, "predict"):
            alpha = self.solver.apply_inverse(y)
