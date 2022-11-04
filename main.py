import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

T = 500
mnorm = tfd.MultivariateNormalDiag([0, 1], [1.5, 1.5])
noise = tfd.Normal(0, .05).sample(T * 2)
noise = tf.reshape(noise, (-1, 2))
positions = np.array([[-1., -1.]], dtype=np.float32)
alpha = 0.3
for t in range(1, T):
    pos_t0 = tf.constant(positions[t - 1][None, :])
    with tf.GradientTape() as tape:
        tape.watch(pos_t0)
        z = mnorm.prob(pos_t0)
        grad = tape.gradient(z, pos_t0)
    pos_t1 = pos_t0 + alpha * grad + noise[t]
    positions = np.vstack([positions, pos_t1.numpy()])

max_pos, min_pos = np.max(positions), np.min(positions)
x = tf.linspace(min_pos, max_pos, 100)
x = tf.cast(x, tf.float32)
xx, yy = tf.meshgrid(x, x)
xy = tf.concat([tf.reshape(xx, (-1, 1)), tf.reshape(yy, (-1, 1))], -1)
with tf.GradientTape() as tape:
    tape.watch(xy)
    z = mnorm.prob(xy)
    grad = tape.gradient(z, xy)

color = cm.get_cmap("inferno")(np.arange(T))
zz = tf.reshape(z, (100, 100))
grad_x, grad_y = grad[:, 0], grad[:, 1]
grad_xx, grad_yy = tf.reshape(grad_x, (100, 100)), tf.reshape(grad_y, (100, 100))
plt.contourf(xx, yy, zz)
plt.quiver(xx[::10, ::10], yy[::10, ::10], grad_xx[::10, ::10], grad_yy[::10, ::10], width=0.003)
plt.scatter(*positions[0], color="red")
for i in range(T):
    plt.plot(positions[i:i+2, 0], positions[i:i+2, 1], c=color[i])
plt.show()
