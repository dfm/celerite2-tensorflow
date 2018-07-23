# CeleriteFlow

A custom TensorFlow Op that implements the celerite solver from [dfm/celerite](https://github.com/dfm/celerite).

## Installation

You'll first need to install [TensorFlow](https://www.tensorflow.org/).
Then:

```bash
git clone https://github.com/dfm/celeriteflow.git
cd celeriteflow
python setup.py install
```

## Usage

Here's a sketch of how you might use this:

```python
import numpy as np
import tensorflow as tf

import celeriteflow as cf

T = tf.float64

np.random.seed(42)
N = 5000
x = tf.constant(np.sort(np.random.uniform(0, 100, N)))
diag = tf.constant(np.random.uniform(0.001, 0.01, N))
y = tf.sin(x)

log_S0 = tf.Variable(0.0, dtype=T)
log_w0 = tf.Variable(0.0, dtype=T)
log_Q = tf.Variable(0.0, dtype=T)

kernel = cf.terms.SHOTerm(log_S0=log_S0,
                          log_w0=log_w0,
                          log_Q=log_Q)

gp = cf.GaussianProcess(kernel, x, y, diag)

loglike = gp.log_likelihood
grad_loglike = tf.gradients(loglike, [log_S0, log_w0, log_Q])
```

You can also call the Cholesky solver directly:

```python
solver = cf.Solver(kernel, x, diag)
alpha = solver.apply_inverse(y[:, None])
logdet = solver.log_determinant
```

Or drop even lower:

```python
d, W = cf.factor(a, U, V, P)
alpha = cf.solve(U, P, d, W, y)
```

where all of the matrices are defined in [Foreman-Mackey (2018)](https://arxiv.org/abs/1801.10156).
