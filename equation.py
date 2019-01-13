import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


class Equation(object):
    """Base class for defining PDE related function."""
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = (self.total_time + 0.0) / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.eigen = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, x, y, z):
        """Generator function in the PDE."""
        return self.eigen * y

    def g_tf(self, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class LaplacianEigen(Equation):
    # eigenvalue problem for Laplacian operator on squares [0, 2pi]^d
    def __init__(self, eqn_config):
        super(LaplacianEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.eigen_array = np.reshape(np.array(eqn_config.eigen_array), [1, self.dim])
        self.eigen = np.sum(self.eigen_array ** 2)
        # self._const_fac = np.sqrt(2.0) ** self._dim
        self.const_fac = 1 / ((np.pi / 2) ** self.dim)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.random.uniform(0.0, np.pi, size=[num_sample, self.dim])
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def g_tf(self, x):
        return tf.reduce_prod(tf.sin(self.eigen_array * x), axis=1, keepdims=True) * self.const_fac

    def true_z(self, x):
        prod = tf.reduce_prod(tf.sin(self.eigen_array * x), axis=1, keepdims=True) * self.const_fac
        return prod / tf.tan(self.eigen_array * x) * self.eigen_array * self.sigma
