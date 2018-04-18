# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["to_dense", "factor", "solve", "get_matrices"]

import os
import sysconfig

import tensorflow as tf

suffix = sysconfig.get_config_var("EXT_SUFFIX")
dirname = os.path.dirname(os.path.abspath(__file__))
mod = tf.load_op_library(os.path.join(dirname, "celerite_op" + suffix))


def to_dense(*args, **kwargs):
    return mod.celerite_to_dense(*args, **kwargs)


def matmul(*args, **kwargs):
    return mod.celerite_mat_mul(*args, **kwargs)


def factor(*args, **kwargs):
    return mod.celerite_factor(*args, **kwargs)[:-1]


def solve(*args, **kwargs):
    return mod.celerite_solve(*args, **kwargs)[0]


@tf.RegisterGradient("CeleriteFactor")
def _celerite_factor_grad(op, *grads):
    args = [op.inputs[1], op.inputs[3]] + list(op.outputs) + list(grads[:-1])
    return mod.celerite_factor_grad(*args)


@tf.RegisterGradient("CeleriteSolve")
def _celerite_solve_grad(op, *grads):
    args = op.inputs[:-1] + list(op.outputs) + list(grads[:-2])
    return mod.celerite_solve_grad(*args)


def get_matrices(a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, diag):
    a = tf.add(diag, tf.reduce_sum(a_real) + tf.reduce_sum(a_comp), name="a")

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
