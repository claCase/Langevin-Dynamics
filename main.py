import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import argparse

k = tf.keras
layers, models, losses, activations, initializers = k.layers, k.models, k.losses, k.activations, k.initializers


class t_log_sigmoid(layers.Layer):
    def __init__(self, initializer=initializers.GlorotNormal()):
        super(t_log_sigmoid, self).__init__()
        self.initializer = initializer

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],), initializer=self.initializer)

    def call(self, inputs):
        return -tf.math.log(1 - self.w * (1 - tf.math.exp(-inputs)))


T = 300
sigma = 0.01
N = 10
mnorm = tfd.MultivariateNormalDiag([0, 1], [1.5, 1.5])
initial_positions = tfd.Uniform(-1, 1).sample(N * 2)
initial_positions = tf.reshape(initial_positions, (N, 2))
noise = tfd.Normal(0, sigma).sample(N * T * 2)
noise = tf.reshape(noise, (N, T, 2))
positions = np.empty(shape=(N, T, 2), dtype=np.float32)
positions[:, 0, :] = initial_positions.numpy()
alpha = 0.2
for t in range(1, T):
    pos_t0 = tf.constant(positions[:, t - 1])
    with tf.GradientTape() as tape:
        tape.watch(pos_t0)
        z = mnorm.prob(pos_t0)
        grad = tape.gradient(z, pos_t0)
    pos_t1 = pos_t0 + alpha * grad + noise[:, t]
    positions[:, t, :] = pos_t1.numpy()


max_pos, min_pos = np.max(positions), np.min(positions)
x = tf.linspace(min_pos * 1.2, max_pos * 1.2, 100)
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


def plot_tot_traj(T):
    plt.cla()
    alphas = np.linspace(0.01, 1, T)
    plt.contourf(xx, yy, zz)
    plt.quiver(xx[::10, ::10], yy[::10, ::10], grad_xx[::10, ::10], grad_yy[::10, ::10], width=0.003)
    #plt.scatter(positions[:, 0, 0], positions[:, 0, 1], color="red")
    for n in range(N):
        for t in range(T):
            plt.plot(positions[n, t:t + 2, 0], positions[n, t:t + 2, 1], c=color[t], alpha=alphas[t])
    plt.scatter(positions[:, T, 0], positions[:, T, 1], color=color[T])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    save = args.save

    fig = plt.figure(figsize=(10, 10))
    anim = animation.FuncAnimation(fig, plot_tot_traj, frames=T - 1, interval=10)
    if save:
        anim.save("./Figures/anim1.gif")
    plt.show()
