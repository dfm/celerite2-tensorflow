import numpy as np
import tensorflow as tf
from celeriteflow.ops import celerite_factor

N = 10
J_real = 2
J_comp = 3

np.random.seed(42)

x = tf.convert_to_tensor(np.sort(np.random.uniform(0, 10, N)), dtype=tf.float32)
diag = tf.convert_to_tensor(np.random.uniform(0.5, 1.5, N), dtype=tf.float32)

a_real = tf.convert_to_tensor(np.random.rand(J_real), dtype=tf.float32)
c_real = tf.convert_to_tensor(np.random.rand(J_real), dtype=tf.float32)

a_comp = tf.convert_to_tensor(np.random.rand(J_comp), dtype=tf.float32)
b_comp = tf.convert_to_tensor(np.random.rand(J_comp), dtype=tf.float32)
c_comp = tf.convert_to_tensor(np.random.rand(J_comp), dtype=tf.float32)
d_comp = tf.convert_to_tensor(np.random.rand(J_comp), dtype=tf.float32)

A = diag + tf.reduce_sum(a_real) + tf.reduce_sum(a_comp)

U = tf.concat((
    a_real + tf.zeros((N, J_real)),
    a_comp[None, :] * tf.cos(d_comp[None, :] * x[:, None])
    + b_comp[None, :] * tf.sin(d_comp[None, :] * x[:, None]),
    a_comp[None, :] * tf.sin(d_comp[None, :] * x[:, None])
    - b_comp[None, :] * tf.cos(d_comp[None, :] * x[:, None]),
), axis=1)

V = tf.concat((
    tf.ones((N, J_real)),
    tf.cos(d_comp[None, :] * x[:, None]),
    tf.sin(d_comp[None, :] * x[:, None]),
), axis=1)

dx = x[1:] - x[:-1]
phi = tf.concat((
    tf.exp(-c_real[None, :] * dx[:, None]),
    tf.exp(-c_comp[None, :] * dx[:, None]),
    tf.exp(-c_comp[None, :] * dx[:, None]),
), axis=1)

factor = celerite_factor(A, U, V, phi)

with tf.Session() as sess:
    D, W, S = sess.run(factor)
    A_n, U_n, V_n, phi_n, x_n = sess.run([A, U, V, phi, x])
    c_real_n, c_comp_n = sess.run([c_real, c_comp])

U_n *= np.concatenate((
    np.exp(-c_real_n[None, :] * x_n[:, None]),
    np.exp(-c_comp_n[None, :] * x_n[:, None]),
    np.exp(-c_comp_n[None, :] * x_n[:, None]),
), axis=1)
V_n *= np.concatenate((
    np.exp(c_real_n[None, :] * x_n[:, None]),
    np.exp(c_comp_n[None, :] * x_n[:, None]),
    np.exp(c_comp_n[None, :] * x_n[:, None]),
), axis=1)
W *= np.concatenate((
    np.exp(c_real_n[None, :] * x_n[:, None]),
    np.exp(c_comp_n[None, :] * x_n[:, None]),
    np.exp(c_comp_n[None, :] * x_n[:, None]),
), axis=1)

L = np.tril(np.dot(U_n, W.T), -1) + np.eye(N)
K2 = np.dot(L, np.dot(np.diag(D), L.T))

K1 = np.tril(np.dot(U_n, V_n.T), -1) + np.triu(np.dot(V_n, U_n.T), 1) + np.diag(A_n)
print(K1 - K2)
