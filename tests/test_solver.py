# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

import celerite
from celerite import terms

import celeriteflow as cf
from celeriteflow import ops


class TestFactor(tf.test.TestCase):

    def test_factor(self, dtype=tf.float64, N=100, Nrhs=1, J_real=2, J_comp=1,
                    seed=123):
        np.random.seed(seed)

        x = np.sort(np.random.uniform(0, 10, N))
        y = np.random.randn(N, Nrhs)
        diag = np.random.uniform(10, 45, N)

        kernel = terms.Term()
        for j in range(J_real):
            kernel += terms.RealTerm(log_a=np.random.uniform(-2, 1),
                                     log_c=np.random.uniform(-2, 1))
        for j in range(J_comp):
            log_a, log_b, log_c = np.random.uniform(-2, 1, 3)
            max_log_d = log_a + log_c - log_b
            kernel += terms.ComplexTerm(log_a=log_a, log_b=log_b, log_c=log_c,
                                        log_d=np.random.uniform(
                                            max_log_d-np.log(2),
                                            max_log_d))

        gp = celerite.GP(kernel)
        gp.compute(x, np.sqrt(diag))
        a_r, c_r, a_c, b_c, c_c, d_c = kernel.coefficients

        # Convert to tensors
        a_real = tf.convert_to_tensor(a_r, dtype=dtype)
        c_real = tf.convert_to_tensor(c_r, dtype=dtype)
        a_comp = tf.convert_to_tensor(a_c, dtype=dtype)
        b_comp = tf.convert_to_tensor(b_c, dtype=dtype)
        c_comp = tf.convert_to_tensor(c_c, dtype=dtype)
        d_comp = tf.convert_to_tensor(d_c, dtype=dtype)
        x_t = tf.convert_to_tensor(x, dtype=dtype)
        y_t = tf.convert_to_tensor(y, dtype=dtype)
        diag_t = tf.convert_to_tensor(diag, dtype=dtype)

        A, U, V, P = cf.get_matrices(
            a_real, c_real, a_comp, b_comp, c_comp, d_comp, x_t, diag_t)
        D, W = ops.factor(A, U, V, P)

        grads = [A, U, V, P]
        shapes = [tuple(map(int, g.shape)) for g in grads]

        log_det = tf.reduce_sum(tf.log(D))
        alpha = ops.solve(U, P, D, W, y_t)
        dotsolve = tf.matmul(alpha, alpha, transpose_a=True)

        with self.test_session() as session:
            A_r, U_r, V_r, P_r = session.run([A, U, V, P])
            D_r, W_r = session.run([D, W])
            log_det_r = session.run(log_det)
            alpha_r = session.run(alpha)
            inits = session.run(grads)

            # Test some gradients
            self.assertAllCloseAccordingToType(
                tf.test.compute_gradient_error(
                    grads, shapes, D, (N,), inits, 1e-8), 0.0)
            self.assertAllCloseAccordingToType(
                tf.test.compute_gradient_error(
                    grads, shapes, log_det, (1,), inits, 1e-6), 0.0)
            self.assertAllCloseAccordingToType(
                tf.test.compute_gradient_error(
                    grads, shapes, W, (N, J_real+2*J_comp), inits, 1e-5), 0.0)
            self.assertAllCloseAccordingToType(
                tf.test.compute_gradient_error(
                    grads, shapes, dotsolve, (Nrhs, Nrhs), inits, 1e-6), 0.0)

        # Test the determinant
        self.assertAllCloseAccordingToType(
            log_det_r, gp.solver.log_determinant())

        alpha0 = gp.apply_inverse(y)
        self.assertAllCloseAccordingToType(alpha0, alpha_r)


if __name__ == "__main__":
    tf.test.main()
