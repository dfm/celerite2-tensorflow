# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Term", "TermSum", "TermProduct", "TermDiff",
    "RealTerm", "ComplexTerm", "SHOTerm", "Matern32Term",
]

import numpy as np
import tensorflow as tf
from itertools import chain
from functools import partial


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
        self.coefficients = self.get_coefficients()

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
            coeffs.append(t.coefficients)
        with tf.name_scope(self.name):
            return [tf.concat(a, 0) for a in zip(*coeffs)]


class TermProduct(Term):

    def __init__(self, term1, term2, **kwargs):
        self.term1 = term1
        self.term2 = term2
        super(TermProduct, self).__init__(**kwargs)

    def get_coefficients(self):
        with tf.name_scope(self.name):
            c1 = self.term1.coefficients
            c2 = self.term2.coefficients

            # First compute real terms
            ar = []
            cr = []
            ar.append(tf.reshape(c1[0][:, None] * c2[0][None, :], [-1]))
            cr.append(tf.reshape(c1[1][:, None] * c2[1][None, :], [-1]))

            # Then the complex terms
            ac = []
            bc = []
            cc = []
            dc = []

            # real * complex
            ac.append(tf.reshape(c1[0][:, None] * c2[2][None, :], [-1]))
            bc.append(tf.reshape(c1[0][:, None] * c2[3][None, :], [-1]))
            cc.append(tf.reshape(c1[1][:, None] + c2[4][None, :], [-1]))
            dc.append(tf.reshape(tf.zeros_like(c1[1])[:, None]
                                 + c2[5][None, :],
                                 [-1]))

            ac.append(tf.reshape(c2[0][:, None] * c1[2][None, :], [-1]))
            bc.append(tf.reshape(c2[0][:, None] * c1[3][None, :], [-1]))
            cc.append(tf.reshape(c2[1][:, None] + c1[4][None, :], [-1]))
            dc.append(tf.reshape(tf.zeros_like(c2[1])[:, None]
                                 + c1[5][None, :],
                                 [-1]))

            # complex * complex
            aj, bj, cj, dj = c1[2:]
            ak, bk, ck, dk = c2[2:]

            ac.append(tf.reshape(
                0.5*(aj[:, None]*ak[None, :] + bj[:, None]*bk[None, :]), [-1]))
            bc.append(tf.reshape(
                0.5*(bj[:, None]*ak[None, :] - aj[:, None]*bk[None, :]), [-1]))
            cc.append(tf.reshape(cj[:, None] + ck[None, :], [-1]))
            dc.append(tf.reshape(dj[:, None] - dk[None, :], [-1]))

            ac.append(tf.reshape(
                0.5*(aj[:, None]*ak[None, :] - bj[:, None]*bk[None, :]), [-1]))
            bc.append(tf.reshape(
                0.5*(bj[:, None]*ak[None, :] + aj[:, None]*bk[None, :]), [-1]))
            cc.append(tf.reshape(cj[:, None] + ck[None, :], [-1]))
            dc.append(tf.reshape(dj[:, None] + dk[None, :], [-1]))

            return [
                tf.concat(vals, axis=0) if len(vals)
                else tf.zeros(0, dtype=self.dtype)
                for vals in (ar, cr, ac, bc, cc, dc)
            ]


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

    def __init__(self, *args, **kwargs):
        super(SHOTerm, self).__init__(*args, **kwargs)

    def get_coefficients(self):
        with tf.name_scope(self.name):
            def overdamped():
                Q = self.Q
                f = tf.sqrt(tf.clip_by_value(4.0*Q**2 - 1.0, 1e-12, np.inf))
                a = self.S0 * self.w0 * Q
                c = 0.5 * self.w0 / Q
                a_size = tf.size(a)
                c_size = tf.size(c)
                return (
                    tf.zeros(0, dtype=self.dtype),
                    tf.zeros(0, dtype=self.dtype),
                    tf.reshape(a, (a_size,)),
                    tf.reshape(a / f, (a_size,)),
                    tf.reshape(c, (c_size,)),
                    tf.reshape(c * f, (c_size,)),
                )

            def underdamped():
                Q = self.Q
                f = tf.sqrt(tf.clip_by_value(1.0 - 4.0*Q**2, 1e-12, np.inf))
                return (
                    0.5*self.S0*self.w0*Q*tf.stack([1.0+1.0/f, 1.0-1.0/f]),
                    0.5*self.w0/Q*tf.stack([1.0-f, 1.0+f]),
                    tf.zeros(0, dtype=self.dtype),
                    tf.zeros(0, dtype=self.dtype),
                    tf.zeros(0, dtype=self.dtype),
                    tf.zeros(0, dtype=self.dtype),
                )

            over = overdamped()
            under = underdamped()

            def true_fn(i):
                return under[i]

            def false_fn(i):
                return over[i]

            m = self.Q < 0.5
            return [
                tf.cond(m, partial(true_fn, i), partial(false_fn, i))
                for i in range(6)]


class Matern32Term(Term):

    parameter_names = ("sigma", "rho")

    def __init__(self, **kwargs):
        dtype = kwargs.get("dtype", tf.float64)
        name = kwargs.get("name", self.__class__.__name__)
        eps = kwargs.pop("eps", None)
        with tf.name_scope(name):
            self.eps = tf.cast(eps, dtype) if eps is not None \
                else tf.constant(0.01, dtype=dtype, name="eps")
        super(Matern32Term, self).__init__(**kwargs)

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
