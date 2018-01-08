import tensorflow as tf
celerite_module = tf.load_op_library('./celerite_op.so')

N = 100

a_real = tf.ones(1)
c_real = tf.ones(1)
a_comp = tf.ones(0)
b_comp = tf.ones(0)
c_comp = tf.ones(0)
d_comp = tf.ones(0)
x = tf.linspace(0.0, 10.0, N)
diag = tf.ones(N)

factor = celerite_module.celerite_factor(
    a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, diag
)

with tf.Session() as sess:
    print(sess.run(factor))
