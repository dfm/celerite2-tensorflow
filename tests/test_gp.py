# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import celeriteflow as cf


def test_gp():
    T = tf.float64
    np.random.seed(42)

    log_S0 = tf.Variable(0.0, dtype=T)
    log_w0 = tf.Variable(0.0, dtype=T)
    log_Q = tf.Variable(0.0, dtype=T)
    kernel = cf.terms.SHOTerm(log_S0=log_S0,
                              log_w0=log_w0,
                              log_Q=log_Q)

    N = 5000

    x = tf.constant(np.sort(np.random.uniform(0, 100, N)))
    diag = tf.constant(np.random.uniform(0.001, 0.01, N))

    y = tf.sin(x)
    gp = cf.GaussianProcess(kernel, x, diag)
    loglike = gp.log_likelihood(y)
    loss = -loglike

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(loss)
        sess.run(tf.gradients(loss, [log_S0, log_w0, log_Q]))

    optimizer_op = tf.train.AdamOptimizer(learning_rate=1e-3)
    minimize_op = optimizer_op.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(minimize_op)
