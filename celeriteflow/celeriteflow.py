# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Solver", "GaussianProcess", "get_matrices"]

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
        self.x = x
        self.diag = diag
        with tf.name_scope(self.name, "Solver"):
            self.a, self.U, self.V, self.P = self.get_matrices()
            self.d, self.W = factor(self.a, self.U, self.V, self.P,
                                    name="factor")
            self.log_determinant = tf.reduce_sum(tf.log(self.d),
                                                 name="log_determinant")
            self.dense = to_dense(self.a, self.U, self.V, self.P,
                                  name="dense")

    def apply_inverse(self, y, **kwargs):
        with tf.name_scope(self.name, "Solver/apply_inverse"):
            return solve(self.U, self.P, self.d, self.W, y, **kwargs)

    def matmul(self, z, **kwargs):
        with tf.name_scope(self.name, "Solver/matmul"):
            return matmul(self.a, self.U, self.V, self.P, z, **kwargs)

    def get_matrices(self, name=None):
        x = self.x
        diag = self.diag
        a_real, c_real, a_comp, b_comp, c_comp, d_comp = \
            self.kernel.get_coefficients()
        if name is None:
            name = "Solver/get_matrices"
        return get_matrices(a_real, c_real, a_comp, b_comp, c_comp, d_comp,
                            x, diag, name=name)


class GaussianProcess(object):

    def __init__(self, kernel, x, diag, name=None):
        with tf.name_scope(name, "GaussianProcess"):
            self.solver = Solver(kernel, x, diag, name=name)

    def log_likelihood(self, y, name=None):
        with tf.name_scope(name, "GaussianProcess/log_likelihood"):
            T = y.dtype
            alpha = self.solver.apply_inverse(y[:, None])
            return -0.5 * (
                tf.squeeze(
                    tf.matmul(y[None, :], alpha))
                + self.solver.log_determinant
                + tf.cast(tf.size(y), T)*tf.constant(np.log(2*np.pi), dtype=T))
