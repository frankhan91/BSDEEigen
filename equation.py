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
        raise NotImplementedError

    def g_tf(self, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class LaplacianEigen(Equation):
    # eigenvalue problem for Laplacian operator on squares [0, 2pi]^d
    def __init__(self, eqn_config):
        super(LaplacianEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample
    
    def f_tf(self, x, y, z):
        shape = tf.shape(x)
        return tf.zeros(shape, tf.float64)
        
    def true_z(self, x):
        shape = tf.shape(x)
        return tf.zeros(shape, tf.float64)

    def true_y(self, x):
        shape = tf.shape(x)
        return tf.ones(shape, tf.float64)


class FokkerPlanckEigen(Equation):
    # eigenvalue problem for Fokker Planck operator on squares [0, 2pi]^d
    def __init__(self, eqn_config):
        super(FokkerPlanckEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        
    def v(self, x):
        # the size of x is (num_sample, dim)
        return tf.cos(x[:,0:1])
    
    def grad_v(self, x):
        temp = -np.sin(x)
        temp[:,1:self.dim] = 0
        return temp
    
    def f_tf(self, x, y, z):
        return -y * tf.cos(x[:,0:1])
#        return y * self.laplician_v(self, x)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        #for now X_0 is uniformly sampled
        x_sample[:, :, 0] = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + \
            self.grad_v(x_sample[:, :, i])*self.delta_t + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample    
    
    def true_y(self, x):
        return tf.exp(-tf.cos(x[:,0:1]))
#        return tf.exp(-self.v(self, x))
        
    def true_z(self, x):
        x1 = tf.concat([x[:,0:1],x[:,1:]*0], axis=1)
        return tf.sin(x1) * tf.exp(-tf.cos(x1)) * self.sigma


class FokkerPlanck2Eigen(Equation):
    # eigenvalue problem for Fokker Planck operator on squares [0, 2pi]^d
    def __init__(self, eqn_config):
        super(FokkerPlanck2Eigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        
    def v(self, x):
        # the size of x is [num_sample, dim]
        return tf.cos(x[:,0:1] + 2 * x[:,1:2])
    
    def f_tf(self, x, y, z):
        return -5 * y * tf.cos(x[:,0:1] + 2 * x[:,1:2])
#        return y * self.laplician_v(self, x)
    
    def grad_v(self, x):
        temp = np.sin(x[:,0:1] + 2 * x[:,1:2])
        grad = np.concatenate([-temp,-2*temp],axis=1)
        if self.dim > 2:
            grad = np.concatenate([grad, x[:,2:]*0],axis=1)
        return grad
    
    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        #for now X_0 is uniformly sampled
        x_sample[:, :, 0] = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + \
            self.grad_v(x_sample[:, :, i])*self.delta_t + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample    
    
    def true_y(self, x):
        return tf.exp(-tf.cos(x[:,0:1] + 2 * x[:,1:2]))
        
    def true_z(self, x):
        temp = tf.sin(x[:,0:1] + 2 * x[:,1:2]) * tf.exp(-tf.cos(x[:,0:1] + 2 * x[:,1:2]))
        z = tf.concat([temp,2*temp],axis=1)
        if self.dim > 2:
            z = tf.concat([z, x[:,2:]*0],axis=1)
        return z * self.sigma


class FokkerPlanck3Eigen(Equation):
    # eigenvalue problem for Fokker Planck operator on squares [0, 2pi]^d
    # v(x) = cos(cos(x1) + 2*x2)
    def __init__(self, eqn_config):
        super(FokkerPlanck3Eigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        
    def v(self, x):
        # the size of x is [num_sample, dim]
        return tf.cos(tf.cos(x[:,0:1]) + 2 * x[:,1:2])
    
    def f_tf(self, x, y, z):
        return y * (tf.cos(x[:,0:1]) * tf.sin(tf.cos(x[:,0:1]) + 2 * x[:,1:2]) - \
                    ( 4 + tf.square(tf.sin(x[:,0:1])) ) * tf.cos(tf.cos(x[:,0:1]) + 2 * x[:,1:2]))
#        return y * self.laplician_v(self, x)
    
    def grad_v(self, x):
        temp = np.sin(np.cos(x[:,0:1]) + 2 * x[:,1:2])
        grad = np.concatenate([np.sin(x[:,0:1])*temp,-2*temp],axis=1)
        if self.dim > 2:
            grad = np.concatenate([grad, x[:,2:]*0],axis=1)
        return grad
    
    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        #for now X_0 is uniformly sampled
        x_sample[:, :, 0] = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + \
            self.grad_v(x_sample[:, :, i])*self.delta_t + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample    
    
    def true_y(self, x):
        return tf.exp(-tf.cos(tf.cos(x[:,0:1]) + 2 * x[:,1:2]))
        
    def true_z(self, x):
        temp = tf.sin(tf.cos(x[:,0:1]) + 2 * x[:,1:2]) * \
        tf.exp(-tf.cos(tf.cos(x[:,0:1]) + 2 * x[:,1:2]))
        z = tf.concat([-tf.sin(x[:,0:1])*temp,2*temp],axis=1)
        if self.dim > 2:
            z = tf.concat([z, x[:,2:]*0],axis=1)
        return z * self.sigma


class HarmonicOscillatorEigen(Equation):
    # eigenvalue problem for Harmonic Oscillator
    def __init__(self, eqn_config):
        super(HarmonicOscillatorEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.random.normal(0.0, 1.0, size=[num_sample, self.dim])
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample
    
    def f_tf(self, x, y, z):
        return -y * tf.reduce_sum(x ** 2, axis=1, keepdims=False)
        
    def true_z(self, x):
        return -x * tf.exp(-0.5 * tf.reduce_sum(x ** 2, axis=1, keepdims=False))

    def true_y(self, x):
        return tf.exp(-0.5 * tf.reduce_sum(x ** 2, axis=1, keepdims=False))