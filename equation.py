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
        self.true_eigen = 0

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
        self.true_eigen = 0
        
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
        self.true_eigen = 0
        
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
        self.true_eigen = 0
        
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


class PotentialEigen(Equation):
    # potential V(x)=cos(x1) on squares [0, 2pi]^d
    def __init__(self, eqn_config):
        super(PotentialEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.true_eigen = -0.378489221264130

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
        return -tf.cos(x[:,0:1])*y

    def true_y(self, x):
        coef = -tf.constant([-0.880348154038233, 0.666404574526491, -0.0765667378952856,\
                0.00408869863723629, -0.000124894301350180, 2.46129968618575e-06,\
                -3.38337623069477e-08, 3.42621149877335e-10, -2.66107082292485e-12], dtype=tf.float64)
        N = 9
        bases = []
        for m in range(N):
            bases += [tf.cos(m * x[:,0:1])]
        bases = tf.concat(bases, axis=1)
        return tf.reduce_sum(coef * bases, axis=1, keepdims=True)
    
    def true_z(self, x):
        coef2 = tf.constant([0, 0.666404574526491, -0.153133475790571, 0.0122660959117089,\
                             -0.000499577205400720, 1.23064984309287e-05, -2.03002573841686e-07,\
                             2.39834804914135e-09, -2.12885665833988e-11], dtype=tf.float64)
        N = 9
        bases = []
        for m in range(N):
            bases += [tf.sin(m * x[:,0:1])]
        bases = tf.concat(bases, axis=1)
        temp = tf.reduce_sum(coef2 * bases, axis=1, keepdims=True)
        return tf.concat([temp,x[:,1:]*0], axis=1) * self.sigma




class SchrodingerEigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d
    def __init__(self, eqn_config):
        super(SchrodingerEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
#        self.ci = [0.814723686393179,0.905791937075619,0.126986816293506,\
#                   0.913375856139019,0.632359246225410,0.097540404999410,\
#                   0.278498218867048,0.546881519204984,0.957506835434298,0.964888535199277]
        self.ci = [0.814723686393179,0.905791937075619]
        self.coef = [[0.904929598872363, 0.892479070097153],
                     [-0.599085866194182, -0.634418280724547],
                     [0.0573984887007387, 0.0668213136994578],
                     [-0.00252519085534048, -0.00325082263430659],
                     [6.32514967687960e-05, 9.02475797129487e-05],
                     [-1.01983519526066e-06, -1.61448458844806e-06],
                     [1.14553116486126e-08, 2.01332109031048e-08],
                     [-9.47170798515555e-11, -1.84883101478958e-10],
                     [6.00357997713989e-13, 1.30180750716843e-12],
                     [-3.00925873281827e-15, -7.24995760563704e-15]]
        
        self.coef2 = [[0, 0],
                      [-0.599085866194182, -0.634418280724547],
                      [0.114796977401477, 0.133642627398916],
                      [-0.00757557256602144, -0.00975246790291976],
                      [0.000253005987075184, 0.000360990318851795],
                      [-5.09917597630331e-06, -8.07242294224032e-06],
                      [6.87318698916753e-08, 1.20799265418629e-07],
                      [-6.63019558960889e-10, -1.29418171035271e-09],
                      [4.80286398171191e-12, 1.04144600573475e-11],
                      [-2.70833285953644e-14, -6.52496184507333e-14]]
        #self.true_eigen = -1.986050602989757
        self.true_eigen = -0.591624518674115
        
        
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
#        bases = []
#        for i in range(self.dim):
#            bases += [tf.cos(x[:,i:i+1])]
#        bases = tf.concat(bases, axis=1)
#        return tf.reduce_sum(self.ci * bases, axis=1, keepdims=True) * y
        return tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y
    
    def true_y(self, x):
        N = 10
        bases_cos = 0 * x
        for m in range(N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def true_z(self, x):
        N = 10
        bases_cos = 0
        for m in range(N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(N):
            bases_sin = bases_sin + tf.sin(m * x) * self.coef2[m] #Broadcasting
        y = tf.reduce_prod(bases_cos, axis=1, keepdims=True)
        #return bases_cos * self.sigma
        return - y * bases_sin / bases_cos * self.sigma



#class HarmonicOscillatorEigen(Equation):
#    # eigenvalue problem for Harmonic Oscillator
#    def __init__(self, eqn_config):
#        super(HarmonicOscillatorEigen, self).__init__(eqn_config)
#        self.sigma = np.sqrt(2.0)
#
#    def sample(self, num_sample):
#        dw_sample = normal.rvs(size=[num_sample,
#                                     self.dim,
#                                     self.num_time_interval]) * self.sqrt_delta_t
#        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
#        x_sample[:, :, 0] = np.random.normal(0.0, 1.0, size=[num_sample, self.dim])
#        for i in range(self.num_time_interval):
#            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
#        return dw_sample, x_sample
#    
#    def f_tf(self, x, y, z):
#        return -y * tf.reduce_sum(x ** 2, axis=1, keepdims=False)
#        
#    def true_z(self, x):
#        return -x * tf.exp(-0.5 * tf.reduce_sum(x ** 2, axis=1, keepdims=False))
#
#    def true_y(self, x):
#        return tf.exp(-0.5 * tf.reduce_sum(x ** 2, axis=1, keepdims=False))