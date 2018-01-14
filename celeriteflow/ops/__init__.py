# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["celerite_factor", "celerite_solve"]

import os
import sysconfig

import tensorflow as tf

suffix = sysconfig.get_config_var("EXT_SUFFIX")
dirname = os.path.dirname(os.path.abspath(__file__))
mod = tf.load_op_library(os.path.join(dirname, "celerite_op" + suffix))

celerite_factor = mod.celerite_factor
celerite_solve = mod.celerite_solve


@tf.RegisterGradient("CeleriteFactor")
def _celerite_factor_grad(op, *grads):
    args = [op.inputs[1], op.inputs[3]] + list(op.outputs) + list(grads)
    return mod.celerite_factor_grad(*args)
