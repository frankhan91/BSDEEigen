import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self._dim = eqn_config.dim
        self._total_time = eqn_config.total_time
        self._num_time_interval = eqn_config.num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._eigen = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, x, y, z):
        """Generator function in the PDE."""
        return self._eigen * y

    def g_tf(self, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t


class LaplacianEigen(Equation):
    def __init__(self, eqn_config):
        super(LaplacianEigen, self).__init__(eqn_config)
        self._sigma = np.sqrt(2.0)
        self._eigen_array = np.reshape(np.array(eqn_config.eigen_array), [1, self._dim])
        self._eigen = np.sum(self._eigen_array**2)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.random.uniform(0.0, np.pi, size=[num_sample, self._dim])
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def g_tf(self, x):
        return tf.reduce_prod(tf.sin(self._eigen_array*x), axis=1, keepdims=True) / (np.pi/2) ** self._dim

    def true_z(self, x):
        prod = tf.reduce_prod(tf.sin(self._eigen_array*x), axis=1, keepdims=True) / (np.pi/2) ** self._dim
        return prod / tf.sin(self._eigen_array*x) * tf.cos(self._eigen_array*x) * self._eigen_array * self._sigma
