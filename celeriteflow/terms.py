# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Term", "TermSum", "TermProduct", "TermDiff",
    "RealTerm", "ComplexTerm", "SHOTerm", "Matern32Term",
]

import numpy as np
import tensorflow as tf
from itertools import chain, product


class Term(object):

    parameter_names = tuple()

    def __init__(self, **kwargs):
        self.dtype = kwargs.pop("dtype", tf.float64)
        self.name = kwargs.pop("name", self.__class__.__name__)
        with tf.name_scope(self.name):
            for name in self.parameter_names:
                if name not in kwargs and "log_" + name not in kwargs:
                    raise ValueError(("Missing required parameter {0}. "
                                      "Provide {0} or log_{0}").format(name))
                value = kwargs[name] if name in kwargs \
                    else tf.exp(kwargs["log_" + name], name=name)
                setattr(self, name, tf.cast(value, self.dtype))

    def __add__(self, b):
        return TermSum(self, b)

    def __radd__(self, b):
        return TermSum(b, self)

    def __mul__(self, b):
        return TermProduct(self, b)

    def __rmul__(self, b):
        return TermProduct(b, self)

    def get_real_coefficients(self):
        with tf.name_scope(self.name):
            return (tf.constant([], dtype=self.dtype, name="a_r"),
                    tf.constant([], dtype=self.dtype, name="c_r"))

    def get_complex_coefficients(self):
        with tf.name_scope(self.name):
            return (tf.constant([], dtype=self.dtype, name="a_c"),
                    tf.constant([], dtype=self.dtype, name="b_c"),
                    tf.constant([], dtype=self.dtype, name="c_c"),
                    tf.constant([], dtype=self.dtype, name="d_c"))

    def get_coefficients(self):
        r = self.get_real_coefficients()
        c = self.get_complex_coefficients()
        return list(chain(r, c))


class TermSum(Term):

    def __init__(self, *terms, **kwargs):
        self.terms = terms
        super(TermSum, self).__init__(**kwargs)

    def get_coefficients(self):
        coeffs = []
        for t in self.terms:
            coeffs.append(t.get_coefficients())
        with tf.name_scope(self.name):
            return [tf.concat(a, 0) for a in zip(*coeffs)]


class TermProduct(Term):

    def __init__(self, term1, term2, **kwargs):
        self.term1 = term1
        self.term2 = term2
        super(TermProduct, self).__init__(**kwargs)

    def get_coefficients(self):
        c1 = self.term1.get_coefficients()
        c2 = self.term2.get_coefficients()

        # First compute real terms
        ar = []
        cr = []
        gen = product(zip(c1[0], c1[1]), zip(c2[0], c2[1]))
        for i, ((aj, cj), (ak, ck)) in enumerate(gen):
            ar.append(aj * ak)
            cr.append(cj + ck)

        # Then the complex terms
        ac = []
        bc = []
        cc = []
        dc = []

        # real * complex
        gen = product(zip(c1[0], c1[1]), zip(*(c2[2:])))
        gen = chain(gen, product(zip(c2[0], c2[1]), zip(*(c1[2:]))))
        for i, ((aj, cj), (ak, bk, ck, dk)) in enumerate(gen):
            ac.append(aj * ak)
            bc.append(aj * bk)
            cc.append(cj + ck)
            dc.append(dk)

        # complex * complex
        gen = product(zip(*(c1[2:])), zip(*(c2[2:])))
        for i, ((aj, bj, cj, dj), (ak, bk, ck, dk)) in enumerate(gen):
            ac.append(0.5 * (aj * ak + bj * bk))
            bc.append(0.5 * (bj * ak - aj * bk))
            cc.append(cj + ck)
            dc.append(dj - dk)

            ac.append(0.5 * (aj * ak - bj * bk))
            bc.append(0.5 * (bj * ak + aj * bk))
            cc.append(cj + ck)
            dc.append(dj + dk)

        with tf.name_scope(self.name):
            return list(map(tf.stack, (ar, cr, ac, bc, cc, dc)))


class TermDiff(Term):

    def __init__(self, term, **kwargs):
        self.term = term
        super(TermDiff, self).__init__(**kwargs)

    def get_coefficients(self):
        with tf.name_scope(self.name):
            coeffs = self.term.get_coefficients()
            a, b, c, d = coeffs[2:]
            final_coeffs = [
                -coeffs[0]*coeffs[1]**2,
                coeffs[1],
                a*(d**2 - c**2) + 2*b*c*d,
                b*(d**2 - c**2) - 2*a*c*d,
                c, d,
            ]
            return final_coeffs


class RealTerm(Term):

    parameter_names = ("a", "c")

    def get_real_coefficients(self):
        with tf.name_scope(self.name):
            return (
                tf.expand_dims(self.a, 0),
                tf.expand_dims(self.c, 0)
            )


class ComplexTerm(Term):

    parameter_names = ("a", "b", "c", "d")

    def get_complex_coefficients(self):
        with tf.name_scope(self.name):
            return (
                tf.expand_dims(self.a, 0),
                tf.expand_dims(self.b, 0),
                tf.expand_dims(self.c, 0),
                tf.expand_dims(self.d, 0),
            )


class SHOTerm(Term):

    parameter_names = ("S0", "w0", "Q")

    def get_coefficients(self):
        with tf.name_scope(self.name):
            def true_fn():
                f = tf.sqrt(4.0*tf.square(self.Q) - 1.0)
                a = self.S0 * self.w0 * self.Q
                c = 0.5 * self.w0 / self.Q
                return (
                    tf.constant([], dtype=self.dtype),
                    tf.constant([], dtype=self.dtype),
                    tf.expand_dims(a, 0),
                    tf.expand_dims(a / f, 0),
                    tf.expand_dims(c, 0),
                    tf.expand_dims(c * f, 0),
                )

            def false_fn():
                f = tf.sqrt(1.0 - 4.0*tf.square(self.Q))
                return (
                    0.5*self.S0*self.w0*self.Q*tf.stack([1.0+1.0/f,
                                                         1.0-1.0/f]),
                    0.5*self.w0/self.Q*tf.stack([1.0-f, 1.0+f]),
                    tf.constant([], dtype=self.dtype),
                    tf.constant([], dtype=self.dtype),
                    tf.constant([], dtype=self.dtype),
                    tf.constant([], dtype=self.dtype),
                )

            return tf.cond(self.Q >= 0.5, true_fn, false_fn)


class Matern32Term(Term):

    parameter_names = ("sigma", "rho")

    def __init__(self, **kwargs):
        eps = kwargs.pop("eps", None)
        super(Matern32Term, self).__init__(**kwargs)
        with tf.name_scope(self.name):
            self.eps = tf.cast(eps, self.dtype) if eps is not None \
                else tf.constant(0.01, dtype=self.dtype, name="eps")

    def get_complex_coefficients(self):
        with tf.name_scope(self.name):
            w0 = np.sqrt(3.0) / self.rho
            S0 = tf.square(self.sigma) / w0
            return (
                tf.expand_dims(w0*S0, 0),
                tf.expand_dims(w0*w0*S0/self.eps, 0),
                tf.expand_dims(w0, 0),
                tf.expand_dims(self.eps, 0)
            )
