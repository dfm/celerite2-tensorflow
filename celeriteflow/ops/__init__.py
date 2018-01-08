# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["celerite_factor_op"]

import os
import sysconfig
import tensorflow as tf

suffix = sysconfig.get_config_var("EXT_SUFFIX")
dirname = os.path.dirname(os.path.abspath(__file__))

celerite_factor_op = tf.load_op_library(os.path.join(
    dirname, "celerite_factor_op" + suffix)).celerite_factor
