# -*- coding: utf-8 -*-

__all__ = ["factor"]

import os
import sysconfig

import tensorflow as tf

suffix = sysconfig.get_config_var("EXT_SUFFIX")
dirname = os.path.dirname(os.path.abspath(__file__))
mod = tf.load_op_library(os.path.join(dirname, "op_impl" + suffix))


# def to_dense(*args, **kwargs):
#     return mod.celerite_to_dense(*args, **kwargs)


# def matmul(*args, **kwargs):
#     return mod.celerite_mat_mul(*args, **kwargs)


def factor(*args, **kwargs):
    return mod.celerite_factor(*args, **kwargs)[:-1]


# def solve(*args, **kwargs):
#     return mod.celerite_solve(*args, **kwargs)[0]


@tf.RegisterGradient("CeleriteFactor")
def _celerite_factor_rev(op, *grads):
    args = list(op.inputs) + list(op.outputs) + list(grads[:-1])
    return mod.celerite_factor_rev(*args)


# @tf.RegisterGradient("CeleriteSolve")
# def _celerite_solve_grad(op, *grads):
#     args = op.inputs[:-1] + list(op.outputs) + list(grads[:-2])
#     return mod.celerite_solve_grad(*args)
