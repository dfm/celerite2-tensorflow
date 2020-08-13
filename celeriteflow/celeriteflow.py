# -*- coding: utf-8 -*-
from __future__ import division, print_function

__all__ = ["Solver", "GaussianProcess", "get_matrices"]

import numpy as np
import tensorflow as tf

from .ops import factor, matmul, solve, to_dense


def get_matrices(
    a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, diag, name=None
):
    with tf.name_scope("get_matrices"):

        a = tf.add(
            diag, tf.reduce_sum(a_real) + tf.reduce_sum(a_comp), name="a"
        )

        expand_left = lambda arg: tf.expand_dims(arg, 0)  # NOQA
        expand_right = lambda arg: tf.expand_dims(arg, -1)  # NOQA

        a_real = expand_left(a_real)
        c_real = expand_left(c_real)
        a_comp = expand_left(a_comp)
        b_comp = expand_left(b_comp)
        c_comp = expand_left(c_comp)
        d_comp = expand_left(d_comp)
        x = expand_right(x)

        U = tf.concat(
            (
                a_real + tf.zeros_like(x),
                a_comp * tf.cos(d_comp * x) + b_comp * tf.sin(d_comp * x),
                a_comp * tf.sin(d_comp * x) - b_comp * tf.cos(d_comp * x),
            ),
            axis=1,
            name="U",
        )

        V = tf.concat(
            (
                tf.zeros_like(a_real) + tf.ones_like(x),
                tf.cos(d_comp * x),
                tf.sin(d_comp * x),
            ),
            axis=1,
            name="V",
        )

        dx = x[1:] - x[:-1]
        arg_real = tf.exp(-c_real * dx)
        arg_comp = tf.exp(-c_comp * dx)
        P = tf.concat((arg_real, arg_comp, arg_comp), axis=1, name="P")

        return a, U, V, P


class Solver(object):
    def __init__(self, kernel, x, diag, name=None):
        self.name = name
        self.kernel = kernel
        self.x = x
        self.diag = diag
        with tf.name_scope("Solver"):
            (
                a_real,
                c_real,
                a_comp,
                b_comp,
                c_comp,
                d_comp,
            ) = self.kernel.coefficients
            self.a, self.U, self.V, self.P = get_matrices(
                a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, diag
            )
            self.d, self.W = factor(
                self.a, self.U, self.V, self.P, name="factor"
            )
            self.log_determinant = tf.reduce_sum(
                tf.log(self.d), name="log_determinant"
            )
            self.dense = to_dense(self.a, self.U, self.V, self.P, name="dense")

    def apply_inverse(self, y, **kwargs):
        with tf.name_scope("Solver/apply_inverse"):
            return solve(self.U, self.P, self.d, self.W, y, **kwargs)

    def matmul(self, z, **kwargs):
        with tf.name_scope("Solver/matmul"):
            return matmul(self.a, self.U, self.V, self.P, z, **kwargs)


class GaussianProcess(object):
    def __init__(self, kernel, x, diag, name=None):
        with tf.name_scope("GaussianProcess"):
            self.solver = Solver(kernel, x, diag, name=name)

    def log_likelihood(self, y, name=None):
        with tf.name_scope("GaussianProcess/log_likelihood"):
            T = y.dtype
            alpha = self.solver.apply_inverse(y[:, None])
            return -0.5 * (
                tf.squeeze(tf.matmul(y[None, :], alpha))
                + self.solver.log_determinant
                + tf.cast(tf.size(y), T)
                * tf.constant(np.log(2 * np.pi), dtype=T)
            )
