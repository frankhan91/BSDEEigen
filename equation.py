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
        return y

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
#        self.eigen = tf.get_variable('eigen', shape=[1], dtype=tf.float64, initializer=None, trainable=True)
        self._const_fac = np.sqrt(2.0) ** self.dim
        # self.const_fac = (np.pi / 2) ** (self.dim)

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
#        return 1 + 0.1* tf.reduce_prod(tf.sin(self.eigen_array * x), axis=1, keepdims=True)
        return tf.reduce_prod(tf.sin(self.eigen_array * x), axis=1, keepdims=True)*self._const_fac + \
               0.05*tf.reduce_prod(tf.sin(2*self.eigen_array * x), axis=1, keepdims=True)*self._const_fac
#        return tf.reduce_prod(x * (np.pi - x) * 2 / (np.pi ** 2), axis=1, keepdims=True) * self.const_fac

    def true_z(self, x):
        return 0
#        prod = tf.reduce_prod(tf.sin(self.eigen_array * x), axis=1, keepdims=True) * self.const_fac
#        return prod / tf.tan(self.eigen_array * x) * self.eigen_array * self.sigma

    def true_y(self, x):
        shape = tf.shape(x)
        return tf.ones(shape, tf.float64)
#        return tf.reduce_prod(tf.sin(self.eigen_array * x), axis=1, keepdims=True)*self._const_fac