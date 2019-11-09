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
        self.sigma = 1
        
    def sample_hist(self, num_sample):
        x_sample = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        return x_sample

    def sample_uniform(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        # uniform X0
        x_sample[:, :, 0] = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def sample_general_new(self, num_sample, sample_func):
        # initially sample according to the current nueral network
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        MCsample = np.zeros(shape=[0, self.dim])
        sup = 2
        while MCsample.shape[0] < num_sample:
            x_smp = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
            pdf = np.abs(sample_func(x_smp))
            reject = np.random.uniform(0.0, sup, size=[num_sample, 1])
            idx = np.nonzero(pdf > reject)
            sample_rate = len(idx[0]) / num_sample
            if sample_rate > 0.8:
                sup *= 2
            elif sample_rate < 0.2:
                sup *= 0.5
                MCsample = np.concatenate([MCsample, x_smp[idx[0], :]], axis=0)
            else:
                MCsample = np.concatenate([MCsample, x_smp[idx[0], :]], axis=0)
        x_sample[:, :, 0] = MCsample[0:num_sample]
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample
    
    
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
    
    def f_tf(self, x, y, z):
        return -y * tf.cos(x[:,0:1])

    def true_y(self, x):
        return tf.exp(-tf.cos(x[:,0:1]))
        
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
    
    def grad_v(self, x):
        temp = np.sin(x[:,0:1] + 2 * x[:,1:2])
        grad = np.concatenate([-temp,-2*temp],axis=1)
        if self.dim > 2:
            grad = np.concatenate([grad, x[:,2:]*0],axis=1)
        return grad

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
        return tf.cos(tf.cos(x[:,0:1]) + 2 * x[:,1:2])
    
    def f_tf(self, x, y, z):
        return y * (tf.cos(x[:,0:1]) * tf.sin(tf.cos(x[:,0:1]) + 2 * x[:,1:2]) - \
                    ( 4 + tf.square(tf.sin(x[:,0:1])) ) * tf.cos(tf.cos(x[:,0:1]) + 2 * x[:,1:2]))
    
    def grad_v(self, x):
        temp = np.sin(np.cos(x[:,0:1]) + 2 * x[:,1:2])
        grad = np.concatenate([np.sin(x[:,0:1])*temp,-2*temp],axis=1)
        if self.dim > 2:
            grad = np.concatenate([grad, x[:,2:]*0],axis=1)
        return grad

    def true_y(self, x):
        return tf.exp(-tf.cos(tf.cos(x[:,0:1]) + 2 * x[:,1:2]))
        
    def true_z(self, x):
        temp = tf.sin(tf.cos(x[:,0:1]) + 2 * x[:,1:2]) * \
        tf.exp(-tf.cos(tf.cos(x[:,0:1]) + 2 * x[:,1:2]))
        z = tf.concat([-tf.sin(x[:,0:1])*temp,2*temp],axis=1)
        if self.dim > 2:
            z = tf.concat([z, x[:,2:]*0],axis=1)
        return z * self.sigma


class FPEigen(Equation):
    # eigenvalue problem for Fokker Planck operator on squares [0, 2pi]^d
    # v(x) = sin(\sum_{i=1}^d c_i cos(x_i))   psi = exp(-v)
    def __init__(self, eqn_config):
        super(FPEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.c = eqn_config.coef
        self.true_eigen = 0
    
    def negativegrad_v_np(self, x):
        temp = np.sum(self.c * np.cos(x), axis=1, keepdims=True) #num_sample x 1
        return self.c * np.sin(x) * np.cos(temp)
    
    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        #for now X_0 is uniformly sampled
        x_sample[:, :, 0] = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] - \
            self.negativegrad_v_np(x_sample[:, :, i])*self.delta_t + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample
    
    def f_tf(self, x, y, z):
        temp = tf.reduce_sum(self.c * tf.cos(x), axis=1, keepdims=True) #num_sample x 1
        temp1 = tf.cos(x) * tf.cos(temp)
        temp2 = self.c * (tf.sin(x) ** 2) * tf.sin(temp)
        return -y * tf.reduce_sum(self.c * (temp1 + temp2), axis=1, keepdims=True)
    
    def true_y(self, x):
        temp = tf.reduce_sum(self.c * tf.cos(x), axis=1, keepdims=True) #num_sample x 1
        return tf.exp(-tf.sin(temp))
        
    def true_z(self, x):
        temp = tf.reduce_sum(self.c * tf.cos(x), axis=1, keepdims=True) #num_sample x 1
        return tf.cos(temp) * tf.sin(x) * self.c * tf.exp(-tf.sin(temp)) * self.sigma


class FPUniEigen(Equation):
    # eigenvalue problem for Fokker Planck operator on squares [0, 2pi]^d
    # uniform sampling with no drift term
    # v(x) = sin(\sum_{i=1}^d c_i cos(x_i))   psi = exp(-v)
    def __init__(self, eqn_config):
        super(FPUniEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.c = eqn_config.coef
        self.true_eigen = 0
    
    def f_tf(self, x, y, z):
        temp = tf.reduce_sum(self.c * tf.cos(x), axis=1, keepdims=True) #num_sample x 1
        temp1 = tf.cos(x) * tf.cos(temp)
        temp2 = self.c * (tf.sin(x) ** 2) * tf.sin(temp)
        Laplacian_v = -tf.reduce_sum(self.c * (temp1 + temp2), axis=1, keepdims=True)
        gradient_v = self.c * tf.sin(x) * (-tf.cos(temp))
        return tf.reduce_sum(gradient_v * z / self.sigma, axis=1, keepdims=True) + Laplacian_v * y
        
    def true_y(self, x):
        temp = tf.reduce_sum(self.c * tf.cos(x), axis=1, keepdims=True) #num_sample x 1
        return tf.exp(-tf.sin(temp))
        
    def true_z(self, x):
        temp = tf.reduce_sum(self.c * tf.cos(x), axis=1, keepdims=True) #num_sample x 1
        return tf.cos(temp) * tf.sin(x) * self.c * tf.exp(-tf.sin(temp)) * self.sigma


class FokkerPlanck4Eigen(Equation):
    # eigenvalue problem for Fokker Planck operator on squares [0, 2pi]^d
    # with a potential W(x). The invariant measure for the first two terms of L is e^cos(x1) while 
    # the eigenfunction for L is e^{-cos(x1)}
    def __init__(self, eqn_config):
        super(FokkerPlanck4Eigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.true_eigen = 0
        self.sup = [2.718,1]
        
    def v(self, x):
        # the size of x is (num_sample, dim)
        return tf.cos(x[:,0:1])
    
    def grad_v(self, x):
        temp = -np.sin(x)
        temp[:,1:self.dim] = 0
        return temp
    
    def f_tf(self, x, y, z):
        return -y * tf.cos(x[:,0:1])
#        return y * 0
    
    def sample_general_old(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        for j in range(self.dim):
            MCsample = []
            while len(MCsample) < num_sample:
                # each time add num_sample samples
                x_smp = np.random.uniform(0.0, 2*np.pi, size=[num_sample])
                pdf = np.abs(np.exp(np.cos(x_smp)))
                reject = np.random.uniform(0.0, self.sup[j], size=[num_sample])
                x_smp = x_smp * (np.sign(pdf - reject) + 1) * 0.5
                temp = x_smp[x_smp != 0]
                MCsample.extend(temp)
            x_sample[:, j, 0] = MCsample[0:num_sample]
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.grad_v(x_sample[:, :, i])*self.delta_t + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def true_y(self, x):
        return tf.exp(-tf.cos(x[:,0:1]))
        
    def true_z(self, x):
        x1 = tf.concat([x[:,0:1],x[:,1:]*0], axis=1)
        return tf.sin(x1) * tf.exp(-tf.cos(x1)) * self.sigma


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
    # d=2
    def __init__(self, eqn_config):
        super(SchrodingerEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.ci = [0.814723686393179,0.905791937075619]
        self.N = 10
        self.sup = [1.56400342750522,1.59706136953917]
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
        self.true_eigen = -0.591624518674115

#    def sample_general_new(self, num_sample, sample_func):
#        dw_sample = normal.rvs(size=[num_sample,
#                                     self.dim,
#                                     self.num_time_interval]) * self.sqrt_delta_t
#        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
#        MCsample = np.zeros(shape=[0, self.dim])
#        sup = 1
#        while MCsample.shape[0] < num_sample:
#            x_smp = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
#            pdf = np.abs(sample_func(x_smp))
#            reject = np.random.uniform(0.0, sup, size=[num_sample, 1])
#            idx = np.nonzero(pdf > reject)
#            sample_rate = len(idx[0]) / num_sample
#            if sample_rate > 0.8:
#                sup *= 2
#            elif sample_rate < 0.2:
#                sup *= 0.5
#                MCsample = np.concatenate([MCsample, x_smp[idx[0], :]], axis=0)
#            else:
#                MCsample = np.concatenate([MCsample, x_smp[idx[0], :]], axis=0)
#        x_sample[:, :, 0] = MCsample[0:num_sample]
#        for i in range(self.num_time_interval):
#            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
#        return dw_sample, x_sample

    def sample_general_old(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        for j in range(self.dim):
            MCsample = []
            while len(MCsample) < num_sample:
                # each time add num_sample samples
                x_smp = np.random.uniform(0.0, 2*np.pi, size=[num_sample])
                bases_cos = 0 * x_smp
                for m in range(self.N):
                    bases_cos = bases_cos + np.cos(m * x_smp) * self.coef[m][j]
                pdf = np.abs(bases_cos)
                reject = np.random.uniform(0.0, self.sup[j], size=[num_sample])
                x_smp = x_smp * (np.sign(pdf - reject) + 1) * 0.5
                temp = x_smp[x_smp != 0]
                MCsample.extend(temp)
            x_sample[:, j, 0] = MCsample[0:num_sample]
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample
    
    def f_tf(self, x, y, z):
        return -tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y

    def true_y_np(self, x):
        # x in shape [num_sample, dim]
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m]  # Broadcasting
        return np.prod(bases_cos, axis=1, keepdims=True)

    def true_y(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def true_z(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + tf.sin(m * x) * self.coef2[m] #Broadcasting
        y = tf.reduce_prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma


class Schrodinger2Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d
    def __init__(self, eqn_config):
        super(Schrodinger2Eigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.N = 10
        self.ci = [0.814723686393179,0.905791937075619,0.126986816293506,\
                   0.913375856139019,0.632359246225410,0.097540404999410,\
                   0.278498218867048,0.546881519204984,0.957506835434298,0.964888535199277]
        # supremum of each dimension's eigenfunction's absolute value
        self.sup = [1.56400342750522,1.59706136953917,1.12365661150691,1.59964582497994,1.48429801251911,1.09574523792265,1.25645243657419,1.43922742759836,1.61421652060220,1.61657863431884]
        self.coef = [[0.904929598872363, 0.892479070097153, 0.996047011600309, 0.891473839010153, 0.931658124581625, 0.997649023058051, 0.982280529025825, 0.944649556846450, 0.885723649385784, 0.884778462161165],
                     [-0.599085866194182, -0.634418280724547, -0.125605497450144, -0.637155464900669, -0.512357488495517, -0.0969095482431325, -0.264889328009242, -0.462960196131467, -0.652506034959353, -0.654980719478657],
                     [0.0573984887007386, 0.0668213136994577, 0.00199001952859007, 0.0676069817707047, 0.0389137499036419, 0.00118025466788750, 0.00914049555158614, 0.0306829183857062, 0.0721762094875563, 0.0729397325313142],
                     [-0.00252519085534048, -0.00325082263430658, -1.40271491734589e-05, -0.00331506767448259, -0.00134207463713329, -6.39243615673157e-06, -0.000140854107058429, -0.000919007154831146, -0.00370016361849726, -0.00376642561780699],
                     [6.32514967687960e-05, 9.02475797129485e-05, 5.56371876669565e-08, 9.27770452504068e-05, 2.62423654134247e-05, 1.94793735338997e-08, 1.22305183605055e-06, 1.55782949374877e-05, 0.000108388631038444, 0.000111150892928655],
                     [-1.01983519526066e-06, -1.61448458844806e-06, -1.41256489759314e-10, -1.67334312048648e-06, -3.29635802399548e-07, -3.79928584585742e-11, -6.80228425921455e-09, -1.69495102259451e-07, -2.04729080278141e-06, -2.11528784557100e-06],
                     [1.14553116486126e-08, 2.01332109031047e-08, 2.49070180873992e-13, 2.10393672744256e-08, 2.88136027661028e-09, 5.14616943511569e-14, 2.62845162481860e-11, 1.28269245292752e-09, 2.69656139309183e-08, 2.80726379468963e-08],
                     [-9.47170798515556e-11, -1.84883101478958e-10, -3.22670848516871e-16, -1.94804537710462e-10, -1.85271772034910e-11, -5.12132251946685e-17, -7.46405433786103e-14, -7.13859314209407e-12, -2.61601555334212e-10, -2.74416243017770e-10],
                     [6.00357997713989e-13, 1.30180750716843e-12, 3.20055741153896e-19, 1.38305603550407e-12, 9.12827390668906e-14, 3.90213598890932e-20, 1.62311055314169e-16, 3.04361603796089e-14, 1.94624368648372e-12, 2.05717944603459e-12],
                     [-3.00925873281827e-15, -7.24995760563703e-15, -2.50838986315946e-22, -7.76650738246465e-15, -3.55551904006216e-16, -2.34921193685406e-23, -2.78916493656665e-19, -1.02576252116579e-16, -1.14534279420053e-14, -1.21989378354622e-14]]

        self.coef2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [-0.599085866194182, -0.634418280724547, -0.125605497450144, -0.637155464900669, -0.512357488495517, -0.0969095482431325, -0.264889328009242, -0.462960196131467, -0.652506034959353, -0.654980719478657],
                      [0.114796977401477, 0.133642627398915, 0.00398003905718015, 0.135213963541409, 0.0778274998072837, 0.00236050933577499, 0.0182809911031723, 0.0613658367714124, 0.144352418975113, 0.145879465062628],
                      [-0.00757557256602144, -0.00975246790291974, -4.20814475203766e-05, -0.00994520302344776, -0.00402622391139988, -1.91773084701947e-05, -0.000422562321175286, -0.00275702146449344, -0.0111004908554918, -0.0112992768534210],
                      [0.000253005987075184, 0.000360990318851794, 2.22548750667826e-07, 0.000371108181001627, 0.000104969461653699, 7.79174941355989e-08, 4.89220734420220e-06, 6.23131797499510e-05, 0.000433554524153778, 0.000444603571714619],
                      [-5.09917597630331e-06, -8.07242294224030e-06, -7.06282448796569e-10, -8.36671560243239e-06, -1.64817901199774e-06, -1.89964292292871e-10, -3.40114212960727e-08, -8.47475511297256e-07, -1.02364540139071e-05, -1.05764392278550e-05],
                      [6.87318698916753e-08, 1.20799265418628e-07, 1.49442108524395e-12, 1.26236203646554e-07, 1.72881616596617e-08, 3.08770166106942e-13, 1.57707097489116e-10, 7.69615471756512e-09, 1.61793683585510e-07, 1.68435827681378e-07],
                      [-6.63019558960889e-10, -1.29418171035270e-09, -2.25869593961810e-15, -1.36363176397323e-09, -1.29690240424437e-10, -3.58492576362679e-16, -5.22483803650272e-13, -4.99701519946585e-11, -1.83121088733948e-09, -1.92091370112439e-09],
                      [4.80286398171191e-12, 1.04144600573474e-11, 2.56044592923117e-18, 1.10644482840325e-11, 7.30261912535125e-13, 3.12170879112745e-19, 1.29848844251335e-15, 2.43489283036871e-13, 1.55699494918698e-11, 1.64574355682767e-11],
                      [-2.70833285953644e-14, -6.52496184507332e-14, -2.25755087684351e-21, -6.98985664421818e-14, -3.19996713605594e-15, -2.11429074316866e-22, -2.51024844290998e-18, -9.23186269049210e-16, -1.03080851478047e-13, -1.09790440519160e-13]]
        self.true_eigen = -1.986050602989757
    
    def sample_general_old(self, num_sample):
        # cheat sampling
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        # uniform sampling is not a good idea
        for j in range(self.dim):
            MCsample = []
            while len(MCsample) < num_sample:
                x_smp = np.random.uniform(0.0, 2*np.pi, size=[num_sample])
                bases_cos = 0 * x_smp
                for m in range(self.N):
                    bases_cos = bases_cos + np.cos(m * x_smp) * self.coef[m][j]
                pdf = np.abs(bases_cos)
                reject = np.random.uniform(0.0, self.sup[j], size=[num_sample])
                x_smp = x_smp * (np.sign(pdf - reject) + 1) * 0.5
                temp = x_smp[x_smp != 0]
                MCsample.extend(temp)
            x_sample[:, j, 0] = MCsample[0:num_sample]
        
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample
    
    def f_tf(self, x, y, z):
        return -tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y
       
    def true_y_np(self,x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m]  # Broadcasting
        return np.prod(bases_cos, axis=1, keepdims=True)
    
    def true_y(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def true_z(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + tf.sin(m * x) * self.coef2[m] #Broadcasting
        y = tf.reduce_prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma


class Schrodinger3Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d
    def __init__(self, eqn_config):
        super(Schrodinger3Eigen, self).__init__(eqn_config)
        self.N = 10
        self.sigma = np.sqrt(2.0)
        self.ci = [0.814723686393179,0.905791937075619,0.126986816293506,\
                   0.913375856139019,0.632359246225410]
        self.sup = [1.56400342750522,1.59706136953917,1.12365661150691,1.59964582497994,1.48429801251911]
        self.coef = [[0.904929598872363, 0.892479070097153, 0.996047011600309, 0.891473839010153, 0.931658124581625],
                     [-0.599085866194182, -0.634418280724547, -0.125605497450144, -0.637155464900669, -0.512357488495517],
                     [0.0573984887007386, 0.0668213136994577, 0.00199001952859007, 0.0676069817707047, 0.0389137499036419],
                     [-0.00252519085534048, -0.00325082263430658, -1.40271491734589e-05, -0.00331506767448259, -0.00134207463713329],
                     [6.32514967687960e-05, 9.02475797129485e-05, 5.56371876669565e-08, 9.27770452504068e-05, 2.62423654134247e-05],
                     [-1.01983519526066e-06, -1.61448458844806e-06, -1.41256489759314e-10, -1.67334312048648e-06, -3.29635802399548e-07],
                     [1.14553116486126e-08, 2.01332109031047e-08, 2.49070180873992e-13, 2.10393672744256e-08, 2.88136027661028e-09],
                     [-9.47170798515556e-11, -1.84883101478958e-10, -3.22670848516871e-16, -1.94804537710462e-10, -1.85271772034910e-11],
                     [6.00357997713989e-13, 1.30180750716843e-12, 3.20055741153896e-19, 1.38305603550407e-12, 9.12827390668906e-14],
                     [-3.00925873281827e-15, -7.24995760563703e-15, -2.50838986315946e-22, -7.76650738246465e-15, -3.55551904006216e-16]]
        self.coef2 = [[0, 0, 0, 0, 0],
                      [-0.599085866194182, -0.634418280724547, -0.125605497450144, -0.637155464900669, -0.512357488495517],
                      [0.114796977401477, 0.133642627398915, 0.00398003905718015, 0.135213963541409, 0.0778274998072837],
                      [-0.00757557256602144, -0.00975246790291974, -4.20814475203766e-05, -0.00994520302344776, -0.00402622391139988],
                      [0.000253005987075184, 0.000360990318851794, 2.22548750667826e-07, 0.000371108181001627, 0.000104969461653699],
                      [-5.09917597630331e-06, -8.07242294224030e-06, -7.06282448796569e-10, -8.36671560243239e-06, -1.64817901199774e-06],
                      [6.87318698916753e-08, 1.20799265418628e-07, 1.49442108524395e-12, 1.26236203646554e-07, 1.72881616596617e-08],
                      [-6.63019558960889e-10, -1.29418171035270e-09, -2.25869593961810e-15, -1.36363176397323e-09, -1.29690240424437e-10],
                      [4.80286398171191e-12, 1.04144600573474e-11, 2.56044592923117e-18, 1.10644482840325e-11, 7.30261912535125e-13],
                      [-2.70833285953644e-14, -6.52496184507332e-14, -2.25755087684351e-21, -6.98985664421818e-14, -3.19996713605594e-15]]
        self.true_eigen = -1.099916247175464
        
    def sample_general_old(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        #x_sample[:, :, 0] = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        for j in range(self.dim):
            MCsample = []
            while len(MCsample) < num_sample:
                x_smp = np.random.uniform(0.0, 2*np.pi, size=[num_sample])
                bases_cos = 0 * x_smp
                for m in range(self.N):
                    bases_cos = bases_cos + np.cos(m * x_smp) * self.coef[m][j]
                pdf = np.abs(bases_cos) #value of function f(x_smp)
                reject = np.random.uniform(0.0, self.sup[j], size=[num_sample])
                x_smp = x_smp * (np.sign(pdf - reject) + 1) * 0.5
                temp = x_smp[x_smp != 0]
                MCsample.extend(temp)
            x_sample[:, j, 0] = MCsample[0:num_sample]
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample
    
    def f_tf(self, x, y, z):
        return -tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y
       
    def true_y_np(self,x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m]  # Broadcasting
        return np.prod(bases_cos, axis=1, keepdims=True)
    
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
        return - y * bases_sin / bases_cos * self.sigma


class Sdg2Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d ci~[0,0.2]
    def __init__(self, eqn_config):
        super(Sdg2Eigen, self).__init__(eqn_config)
        self.N = 10
        self.sigma = np.sqrt(2.0)
        self.ci = [0.162944737278636,0.181158387415124,0.025397363258701,0.182675171227804,0.126471849245082,0.019508080999882,0.055699643773410,0.109376303840997,0.191501367086860,0.192977707039855]
        self.coef = [[0.993571793881952,0.992112802401298,0.999838871758632,0.991985354482078,0.996078392055002,0.999904903363006,0.999227344468429,0.997052622938466,0.991226051652424,0.991096131179570],
                     [-0.160061291194770,-0.177224122143104,-0.0253861093252471,-0.178643778619933,-0.125107059922130,-0.0195029792507090,-0.0555812781968171,-0.108489355137516,-0.186874250374073,-0.188245776894456],
                     [0.00325007727123395,0.00399794197948296,8.05863952861287e-05,0.00406345319436995,0.00197411443935602,4.75560761524560e-05,0.000386840691309466,0.00148119005866490,0.00445436426915333,0.00452135628602182],
                     [-2.93797849166571e-05,-4.01667801745879e-05,-1.13700607076658e-07,-4.11655916879119e-05,-1.38587047043284e-05,-5.15393770949078e-08,-1.19684981038921e-06,-8.99463377780295e-06,-4.72979651314309e-05,-4.83779988050594e-05],
                     [1.49482399351235e-07,2.27167087775217e-07,9.02407255701227e-11,2.34761407474919e-07,5.47463713973189e-08,3.14199988139629e-11,2.08305573451501e-09,3.07325435388975e-08,2.82738262076159e-07,2.91419081794217e-07],
                     [-4.86883641624589e-10,-8.22539598941590e-10,-4.58385880437081e-14,-8.57145589811237e-10,-1.38431507219819e-10,-1.22591238099021e-14,-2.32034750394283e-12,-6.72113561115208e-11,-1.08212492344907e-09,-1.12393541893816e-09],
                     [1.10142399609609e-12,2.06867289930651e-12,1.61696832610095e-17,2.17373950410335e-12,2.43099654874719e-13,3.32164395266348e-18,1.79492650923155e-15,1.02081589368955e-13,2.87676001107095e-12,3.01092258164098e-12],
                     [-1.83072364364542e-15,-3.82285292819827e-15,-4.19063409950640e-21,-4.05062488897179e-15,-3.13659344249419e-16,-6.61232423279210e-22,-1.02011344818844e-18,-1.13912481483233e-16,-5.61948722482390e-15,-5.92687179290964e-15],
                     [2.32985123369096e-18,5.40924038129660e-18,8.31522997654433e-25,5.77949723343268e-18,3.09855992670163e-19,1.00779299219574e-25,4.43881934470464e-22,9.73238990068554e-20,8.40517382052742e-18,8.93324045857629e-18],
                     [-2.34284521622012e-21,-6.04791979155140e-21,-1.30365676047138e-28,-6.51597707320250e-21,-2.41860537516298e-22,-1.21362085143571e-29,-1.52609785500994e-25,-6.57005000646960e-23,-9.93391805631913e-21,-1.06393915310827e-20]]
        self.coef2 = [[0,0,0,0,0,0,0,0,0,0],
                      [-0.160061291194770,-0.177224122143104,-0.0253861093252471,-0.178643778619933,-0.125107059922130,-0.0195029792507090,-0.0555812781968171,-0.108489355137516,-0.186874250374073,-0.188245776894456],
                      [0.00650015454246790,0.00799588395896592,0.000161172790572257,0.00812690638873990,0.00394822887871203,9.51121523049121e-05,0.000773681382618932,0.00296238011732981,0.00890872853830666,0.00904271257204364],
                      [-8.81393547499713e-05,-0.000120500340523764,-3.41101821229973e-07,-0.000123496775063736,-4.15761141129853e-05,-1.54618131284724e-07,-3.59054943116762e-06,-2.69839013334089e-05,-0.000141893895394293,-0.000145133996415178],
                      [5.97929597404939e-07,9.08668351100869e-07,3.60962902280491e-10,9.39045629899676e-07,2.18985485589276e-07,1.25679995255852e-10,8.33222293806002e-09,1.22930174155590e-07,1.13095304830463e-06,1.16567632717687e-06],
                      [-2.43441820812295e-09,-4.11269799470795e-09,-2.29192940218540e-13,-4.28572794905618e-09,-6.92157536099094e-10,-6.12956190495104e-14,-1.16017375197142e-11,-3.36056780557604e-10,-5.41062461724537e-09,-5.61967709469082e-09],
                      [6.60854397657655e-12,1.24120373958391e-11,9.70180995660572e-17,1.30424370246201e-11,1.45859792924832e-12,1.99298637159809e-17,1.07695590553893e-14,6.12489536213729e-13,1.72605600664257e-11,1.80655354898459e-11],
                      [-1.28150655055179e-14,-2.67599704973879e-14,-2.93344386965448e-20,-2.83543742228025e-14,-2.19561540974593e-15,-4.62862696295447e-21,-7.14079413731909e-18,-7.97387370382634e-16,-3.93364105737673e-14,-4.14881025503675e-14],
                      [1.86388098695277e-17,4.32739230503728e-17,6.65218398123546e-24,4.62359778674614e-17,2.47884794136130e-18,8.06234393756592e-25,3.55105547576371e-21,7.78591192054843e-19,6.72413905642194e-17,7.14659236686104e-17],
                      [-2.10856069459811e-20,-5.44312781239626e-20,-1.17329108442424e-27,-5.86437936588225e-20,-2.17674483764668e-21,-1.09225876629214e-28,-1.37348806950895e-24,-5.91304500582264e-22,-8.94052625068721e-20,-9.57545237797440e-20]]
        self.true_eigen = -0.098087448866409

    def f_tf(self, x, y, z):
        return -tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y
       
    def true_y_np(self,x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m]  # Broadcasting
        return np.prod(bases_cos, axis=1, keepdims=True)
    
    def true_y(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def true_z(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + tf.sin(m * x) * self.coef2[m] #Broadcasting
        y = tf.reduce_prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma


class Sdg3Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d ci~[0,0.2]
    def __init__(self, eqn_config):
        super(Sdg3Eigen, self).__init__(eqn_config)
        self.N = 10
        self.sigma = np.sqrt(2.0)
        self.ci = [0.162944737278636,0.181158387415124,0.025397363258701,0.182675171227804,0.126471849245082]
        self.coef = [[0.993571793881952,0.992112802401298,0.999838871758632,0.991985354482078,0.996078392055002],
                     [-0.160061291194770,-0.177224122143104,-0.0253861093252471,-0.178643778619933,-0.125107059922130],
                     [0.00325007727123395,0.00399794197948296,8.05863952861287e-05,0.00406345319436995,0.00197411443935602],
                     [-2.93797849166571e-05,-4.01667801745879e-05,-1.13700607076658e-07,-4.11655916879119e-05,-1.38587047043284e-05],
                     [1.49482399351235e-07,2.27167087775217e-07,9.02407255701227e-11,2.34761407474919e-07,5.47463713973189e-08],
                     [-4.86883641624589e-10,-8.22539598941590e-10,-4.58385880437081e-14,-8.57145589811237e-10,-1.38431507219819e-10],
                     [1.10142399609609e-12,2.06867289930651e-12,1.61696832610095e-17,2.17373950410335e-12,2.43099654874719e-13],
                     [-1.83072364364542e-15,-3.82285292819827e-15,-4.19063409950640e-21,-4.05062488897179e-15,-3.13659344249419e-16],
                     [2.32985123369096e-18,5.40924038129660e-18,8.31522997654433e-25,5.77949723343268e-18,3.09855992670163e-19],
                     [-2.34284521622012e-21,-6.04791979155140e-21,-1.30365676047138e-28,-6.51597707320250e-21,-2.41860537516298e-22]]
        self.coef2 = [[0,0,0,0,0],
                      [-0.160061291194770,-0.177224122143104,-0.0253861093252471,-0.178643778619933,-0.125107059922130],
                      [0.00650015454246790,0.00799588395896592,0.000161172790572257,0.00812690638873990,0.00394822887871203],
                      [-8.81393547499713e-05,-0.000120500340523764,-3.41101821229973e-07,-0.000123496775063736,-4.15761141129853e-05],
                      [5.97929597404939e-07,9.08668351100869e-07,3.60962902280491e-10,9.39045629899676e-07,2.18985485589276e-07],
                      [-2.43441820812295e-09,-4.11269799470795e-09,-2.29192940218540e-13,-4.28572794905618e-09,-6.92157536099094e-10],
                      [6.60854397657655e-12,1.24120373958391e-11,9.70180995660572e-17,1.30424370246201e-11,1.45859792924832e-12],
                      [-1.28150655055179e-14,-2.67599704973879e-14,-2.93344386965448e-20,-2.83543742228025e-14,-2.19561540974593e-15],
                      [1.86388098695277e-17,4.32739230503728e-17,6.65218398123546e-24,4.62359778674614e-17,2.47884794136130e-18],
                      [-2.10856069459811e-20,-5.44312781239626e-20,-1.17329108442424e-27,-5.86437936588225e-20,-2.17674483764668e-21]]
        self.true_eigen = -0.054018930536326

    def f_tf(self, x, y, z):
        return -tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y
       
    def true_y_np(self,x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m]  # Broadcasting
        return np.prod(bases_cos, axis=1, keepdims=True)
    
    def true_y(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def true_z(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + tf.sin(m * x) * self.coef2[m] #Broadcasting
        y = tf.reduce_prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma



class Sdg4Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d ci~[0,0.2] dim=20
    def __init__(self, eqn_config):
        super(Sdg4Eigen, self).__init__(eqn_config)
        self.N = 10
        self.sigma = np.sqrt(2.0)
        self.ci = [0.814723686393179,0.905791937075619,0.126986816293506,0.913375856139019,
                   0.632359246225410,0.0975404049994095,0.278498218867048,0.546881519204984,
                   0.957506835434298,0.964888535199277,0.157613081677548,0.970592781760616,
                   0.957166948242946,0.485375648722841,0.800280468888800,0.141886338627215,
                   0.421761282626275,0.915735525189067,0.792207329559554,0.959492426392903];
        self.coef = [[9.93571793881952e-01,9.92112802401298e-01,9.99838871758632e-01,9.91985354482078e-01,9.96078392055001e-01,9.99904903363006e-01,9.99227344468429e-01,9.97052622938466e-01,9.91226051652424e-01,9.91096131179570e-01,
                     9.99751885431387e-01,9.90995168763630e-01,9.91232013843579e-01,9.97671141961725e-01,9.93790836096261e-01,9.99798882547267e-01,9.98236638945445e-01,9.91945517012674e-01,9.93911770295563e-01,9.91191185937591e-01],
                     [-1.60061291194770e-01,-1.77224122143104e-01,-2.53861093252471e-02,-1.78643778619933e-01,-1.25107059922130e-01,-1.95029792507089e-02,-5.55812781968171e-02,-1.08489355137516e-01,-1.86874250374073e-01,-1.88245776894456e-01,
                     -3.15011071139622e-02,-1.89304601684690e-01,-1.86811062954119e-01,-9.64531914749071e-02,-1.57320372335326e-01,-2.83615725129879e-02,-8.39430951725140e-02,-1.79085181681518e-01,-1.55786138732998e-01,-1.87243322325137e-01],
                     [3.25007727123395e-03,3.99794197948296e-03,8.05863952861287e-05,4.06345319436995e-03,1.97411443935602e-03,4.75560761524556e-05,3.86840691309466e-04,1.48119005866490e-03,4.45436426915333e-03,4.52135628602182e-03,
                     1.24110111641962e-04,4.57343768482075e-03,4.45129067198346e-03,1.16910571275779e-03,3.13812904714650e-03,1.00593432512445e-04,8.84358241255061e-04,4.08393663011324e-03,3.07635875747430e-03,4.47233935578618e-03],
                     [-2.93797849166571e-05,-4.01667801745879e-05,-1.13700607076658e-07,-4.11655916879119e-05,-1.38587047043284e-05,-5.15393770949071e-08,-1.19684981038921e-06,-8.99463377780295e-06,-4.72979651314309e-05,-4.83779988050594e-05,
                     -2.17337017355155e-07,-4.92234301570630e-05,-4.72486147891217e-05,-6.30187780671723e-06,-2.78662727379379e-05,-1.58580172301416e-07,-4.14273047533905e-06,-4.14796151347149e-05,-2.70429111922196e-05,-4.75869318780308e-05],
                     [1.49482399351235e-07,2.27167087775217e-07,9.02407255701227e-11,2.34761407474919e-07,5.47463713973189e-08,3.14199988139623e-11,2.08305573451501e-09,3.07325435388974e-08,2.82738262076159e-07,2.91419081794217e-07,
                     2.14094147924733e-10,2.98260808599963e-07,2.82343213752958e-07,1.91118700343264e-08,1.39272165361798e-07,1.40627294460908e-10,1.09178991676541e-08,2.37162143789208e-07,1.33795716566563e-07,2.85054263313073e-07],
                     [-4.86883641624589e-10,-8.22539598941590e-10,-4.58385880437081e-14,-8.57145589811237e-10,-1.38431507219819e-10,-1.22591238099018e-14,-2.32034750394283e-12,-6.72113561115208e-11,-1.08212492344907e-09,-1.12393541893816e-09,
                     -1.34979489620910e-13,-1.15711310848761e-09,-1.08022990465101e-09,-3.70983326465558e-11,-4.45593975866445e-10,-7.98142887882517e-14,-1.84161754767563e-11,-8.68145177479112e-10,-4.23758317869357e-10,-1.09324817672422e-09],
                     [1.10142399609609e-12,2.06867289930651e-12,1.61696832610095e-17,2.17373950410335e-12,2.43099654874719e-13,3.32164395266338e-18,1.79492650923155e-15,1.02081589368955e-13,2.87676001107095e-12,3.01092258164098e-12,
                     5.90981491244311e-17,3.11810988925862e-12,2.87070383638164e-12,5.00104226564305e-14,9.90162209508122e-13,3.14582095596025e-17,2.15729565032615e-14,2.20731746504828e-12,9.32148527511482e-13,2.91235148049679e-12],
                     [-1.83072364364542e-15,-3.82285292819827e-15,-4.19063409950640e-21,-4.05062488897179e-15,-3.13659344249419e-16,-6.61232423279187e-22,-1.02011344818844e-18,-1.13912481483233e-16,-5.61948722482390e-15,-5.92687179290964e-15,
                     -1.90102835983106e-20,-6.17412510215881e-15,-5.60566789202635e-15,-4.95316136383162e-17,-1.61663137301501e-15,-9.10952252498444e-21,-1.85666171934218e-17,-4.12381467764619e-15,-1.50656859740137e-15,-5.70080089776728e-15],
                     [2.32985123369096e-18,5.40924038129660e-18,8.31522997654433e-25,5.77949723343267e-18,3.09855992670163e-19,1.00779299219570e-25,4.43881934470464e-22,9.73238990068554e-20,8.40517382052742e-18,8.93324045857629e-18,
                     4.68187537497583e-24,9.36089643670863e-18,8.38152930383915e-18,3.75600862379827e-20,2.02093323724025e-18,2.01963844426296e-24,1.22342163043262e-20,5.89911911869808e-18,1.86435474287292e-18,8.54446891500644e-18],
                     [-2.34284521622012e-21,-6.04791979155140e-21,-1.30365676047138e-28,-6.51597707320250e-21,-2.41860537516298e-22,-1.21362085143565e-29,-1.52609785500994e-25,-6.57005000646960e-23,-9.93391805631913e-21,-1.06393915310827e-20,
                     -9.11057104646975e-28,-1.12146068354369e-20,-9.90245814250184e-21,-2.25045408189921e-23,-1.99619218597823e-21,-3.53790463229565e-28,-6.36968049317633e-24,-6.66801812916063e-21,-1.82296064009513e-21,-1.01194812202975e-20]]
        self.coef2 = [[0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,
                      0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00,0.00000000000000e+00],
                      [-1.60061291194770e-01,-1.77224122143104e-01,-2.53861093252471e-02,-1.78643778619933e-01,-1.25107059922130e-01,-1.95029792507089e-02,-5.55812781968171e-02,-1.08489355137516e-01,-1.86874250374073e-01,-1.88245776894456e-01,
                      -3.15011071139622e-02,-1.89304601684690e-01,-1.86811062954119e-01,-9.64531914749071e-02,-1.57320372335326e-01,-2.83615725129879e-02,-8.39430951725140e-02,-1.79085181681518e-01,-1.55786138732998e-01,-1.87243322325137e-01],
                      [6.50015454246790e-03,7.99588395896592e-03,1.61172790572257e-04,8.12690638873990e-03,3.94822887871203e-03,9.51121523049111e-05,7.73681382618932e-04,2.96238011732981e-03,8.90872853830666e-03,9.04271257204364e-03,
                      2.48220223283924e-04,9.14687536964150e-03,8.90258134396692e-03,2.33821142551557e-03,6.27625809429300e-03,2.01186865024891e-04,1.76871648251012e-03,8.16787326022648e-03,6.15271751494860e-03,8.94467871157237e-03],
                      [-8.81393547499713e-05,-1.20500340523764e-04,-3.41101821229973e-07,-1.23496775063736e-04,-4.15761141129853e-05,-1.54618131284721e-07,-3.59054943116762e-06,-2.69839013334089e-05,-1.41893895394293e-04,-1.45133996415178e-04,
                      -6.52011052065466e-07,-1.47670290471189e-04,-1.41745844367365e-04,-1.89056334201517e-05,-8.35988182138136e-05,-4.75740516904248e-07,-1.24281914260172e-05,-1.24438845404145e-04,-8.11287335766587e-05,-1.42760795634092e-04],
                      [5.97929597404939e-07,9.08668351100869e-07,3.60962902280491e-10,9.39045629899676e-07,2.18985485589276e-07,1.25679995255849e-10,8.33222293806002e-09,1.22930174155590e-07,1.13095304830463e-06,1.16567632717687e-06,
                      8.56376591698930e-10,1.19304323439985e-06,1.12937285501183e-06,7.64474801373055e-08,5.57088661447191e-07,5.62509177843630e-10,4.36715966706163e-08,9.48648575156832e-07,5.35182866266250e-07,1.14021705325229e-06],
                      [-2.43441820812294e-09,-4.11269799470795e-09,-2.29192940218540e-13,-4.28572794905618e-09,-6.92157536099094e-10,-6.12956190495089e-14,-1.16017375197142e-11,-3.36056780557604e-10,-5.41062461724537e-09,-5.61967709469082e-09,
                      -6.74897448104548e-13,-5.78556554243806e-09,-5.40114952325507e-09,-1.85491663232779e-10,-2.22796987933223e-09,-3.99071443941258e-13,-9.20808773837813e-11,-4.34072588739556e-09,-2.11879158934679e-09,-5.46624088362109e-09],
                      [6.60854397657655e-12,1.24120373958391e-11,9.70180995660572e-17,1.30424370246201e-11,1.45859792924832e-12,1.99298637159803e-17,1.07695590553893e-14,6.12489536213729e-13,1.72605600664257e-11,1.80655354898459e-11,
                      3.54588894746586e-16,1.87086593355517e-11,1.72242230182898e-11,3.00062535938583e-13,5.94097325704873e-12,1.88749257357615e-16,1.29437739019569e-13,1.32439047902897e-11,5.59289116506889e-12,1.74741088829807e-11],
                      [-1.28150655055179e-14,-2.67599704973879e-14,-2.93344386965448e-20,-2.83543742228025e-14,-2.19561540974593e-15,-4.62862696295431e-21,-7.14079413731909e-18,-7.97387370382634e-16,-3.93364105737673e-14,-4.14881025503675e-14,
                      -1.33071985188174e-19,-4.32188757151117e-14,-3.92396752441845e-14,-3.46721295468214e-16,-1.13164196111051e-14,-6.37666576748911e-20,-1.29966320353952e-16,-2.88667027435233e-14,-1.05459801818096e-14,-3.99056062843709e-14],
                      [1.86388098695277e-17,4.32739230503728e-17,6.65218398123546e-24,4.62359778674614e-17,2.47884794136130e-18,8.06234393756561e-25,3.55105547576371e-21,7.78591192054843e-19,6.72413905642194e-17,7.14659236686103e-17,
                      3.74550029998067e-23,7.48871714936690e-17,6.70522344307132e-17,3.00480689903862e-19,1.61674658979220e-17,1.61571075541037e-23,9.78737304346097e-20,4.71929529495846e-17,1.49148379429834e-17,6.83557513200516e-17],
                      [-2.10856069459811e-20,-5.44312781239626e-20,-1.17329108442424e-27,-5.86437936588225e-20,-2.17674483764668e-21,-1.09225876629209e-28,-1.37348806950895e-24,-5.91304500582264e-22,-8.94052625068721e-20,-9.57545237797440e-20,
                      -8.19951394182277e-27,-1.00931461518932e-19,-8.91221232825165e-20,-2.02540867370929e-22,-1.79657296738041e-20,-3.18411416906609e-27,-5.73271244385870e-23,-6.00121631624457e-20,-1.64066457608562e-20,-9.10753309826775e-20]]
        self.true_eigen = -0.203549513655507

    def f_tf(self, x, y, z):
        return -tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y
       
    def true_y_np(self,x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m]  # Broadcasting
        return np.prod(bases_cos, axis=1, keepdims=True)
    
    def true_y(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def true_z(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + tf.sin(m * x) * self.coef2[m] #Broadcasting
        y = tf.reduce_prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma


class Schrodinger4Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d, only c1 is big
    def __init__(self, eqn_config):
        super(Schrodinger4Eigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.ci = [0.814723686393179,0.009057919370756,0.001269868162935,0.009133758561390,0.006323592462254,9.754040499941000e-04,0.002784982188670,0.005468815192050,0.009575068354343,0.009648885351993]

        self.coef = [[0.904929598872363,0.999979490601231,0.999999596859515,0.999979145761134,0.999990003538014,0.999999762147014,0.999998060987116,0.999992523291119,0.999977082110068,0.999976727427334],
                     [-0.599085866194182,-0.00905740849473516,-0.00126986675511298,-0.00913323474626525,-0.00632341862470508,-0.000975403411986470,-0.00278496733834586,-0.00546870274788088,-0.00957446489110190,-0.00964826782510098],
                     [0.0573984887007387,1.02550601551601e-05,2.01570382062582e-07,1.04274924372526e-05,4.99831669831588e-06,1.18926541445936e-07,9.69509666433935e-07,3.73840238420714e-06,1.14593954473278e-05,1.16367508661293e-05],
                     [-0.00252519085534048,-5.16050543753271e-09,-1.42203324456659e-11,-5.29120948393997e-09,-1.75595832088725e-09,-6.44448866540066e-12,-1.50001350798404e-10,-1.13577664263326e-09,-6.09577518456065e-09,-6.23783957776454e-09],
                     [6.32514967687960e-05,1.46074065479235e-12,5.64306353994688e-16,1.51027808398705e-12,3.47000427769028e-13,1.96435683150549e-16,1.30544915863302e-14,1.94097975097372e-13,1.82399364600819e-12,1.88089197576217e-12],
                     [-1.01983519526066e-06,-2.64628401129313e-16,-1.43317890804571e-20,-2.75893436210482e-16,-4.38861463029835e-17,-3.83206179867279e-21,-7.27118865737387e-19,-2.12290451215048e-17,-3.49301399003363e-16,-3.62974532001616e-16],
                     [1.14553116486126e-08,3.32918853316547e-20,2.52768995963679e-25,3.49997085759293e-20,3.85445615682165e-21,5.19137565872877e-26,2.81247908642715e-23,1.61242412059455e-21,4.64532394288600e-20,4.86437604504759e-20],
                     [-9.47170798515555e-11,-3.07713862774977e-24,-3.27532207633045e-30,-3.26207713771296e-24,-2.48716951786648e-25,-5.16700796440313e-31,-7.99246014257390e-28,-8.99779839259299e-26,-4.53877281804158e-24,-4.78944171223770e-24],
                     [6.00357997713989e-13,2.17756813995297e-28,3.24938119771033e-35,2.32776988102891e-28,1.22875045257916e-29,3.93742386268277e-36,1.73895563522134e-32,3.84424272441425e-30,3.39529024360888e-28,3.61042779065732e-28],
                     [-3.00925873281827e-15,-1.21756210233626e-32,-2.54707977539367e-40,-1.31244326084821e-32,-4.79641489713414e-34,-2.37072061333013e-41,-2.98945550500580e-37,-1.29772153666652e-34,-2.00682743619625e-32,-2.15043884768845e-32]]

        self.coef2 = [[0,0,0,0,0,0,0,0,0,0],
                      [-0.599085866194182,-0.00905740849473516,-0.00126986675511298,-0.00913323474626525,-0.00632341862470508,-0.000975403411986470,-0.00278496733834586,-0.00546870274788088,-0.00957446489110190,-0.00964826782510098],
                      [0.114796977401477,2.05101203103202e-05,4.03140764125165e-07,2.08549848745052e-05,9.99663339663175e-06,2.37853082891872e-07,1.93901933286787e-06,7.47680476841428e-06,2.29187908946556e-05,2.32735017322586e-05],
                      [-0.00757557256602144,-1.54815163125981e-08,-4.26609973369978e-11,-1.58736284518199e-08,-5.26787496266174e-09,-1.93334659962020e-11,-4.50004052395211e-10,-3.40732992789978e-09,-1.82873255536820e-08,-1.87135187332936e-08],
                      [0.000253005987075184,5.84296261916939e-12,2.25722541597875e-15,6.04111233594821e-12,1.38800171107611e-12,7.85742732602196e-16,5.22179663453207e-14,7.76391900389488e-13,7.29597458403274e-12,7.52356790304868e-12],
                      [-5.09917597630331e-06,-1.32314200564656e-15,-7.16589454022856e-20,-1.37946718105241e-15,-2.19430731514917e-16,-1.91603089933640e-20,-3.63559432868694e-18,-1.06145225607524e-16,-1.74650699501681e-15,-1.81487266000808e-15],
                      [6.87318698916753e-08,1.99751311989928e-19,1.51661397578208e-24,2.09998251455576e-19,2.31267369409299e-20,3.11482539523726e-25,1.68748745185629e-22,9.67454472356732e-21,2.78719436573160e-19,2.91862562702856e-19],
                      [-6.63019558960889e-10,-2.15399703942484e-23,-2.29272545343131e-29,-2.28345399639907e-23,-1.74101866250654e-24,-3.61690557508219e-30,-5.59472209980173e-27,-6.29845887481509e-25,-3.17714097262911e-23,-3.35260919856639e-23],
                      [4.80286398171191e-12,1.74205451196238e-27,2.59950495816826e-34,1.86221590482313e-27,9.83000362063329e-29,3.14993909014621e-35,1.39116450817707e-31,3.07539417953140e-29,2.71623219488711e-27,2.88834223252585e-27],
                      [-2.70833285953644e-14,-1.09580589210263e-31,-2.29237179785430e-39,-1.18119893476339e-31,-4.31677340742072e-33,-2.13364855199712e-40,-2.69050995450522e-36,-1.16794938299987e-33,-1.80614469257663e-31,-1.93539496291961e-31]]

        self.true_eigen = -0.269898883873185
               

    def f_tf(self, x, y, z):
        return -tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y
       
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
        return - y * bases_sin / bases_cos * self.sigma


class Schrodinger5Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^2 ci*cos(xi) on squares [0, 2pi]^d
    # d=2 the second smallest eigenvalue,the initialization should be 0
    # dim1 second smallest eigenvalue, dim2 smallest eigenvalue
    def __init__(self, eqn_config):
        super(Schrodinger5Eigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.ci = [0.814723686393179,0.905791937075619]
        self.N = 10
        self.sup = [1.56400342750522,1.59706136953917]
        self.coef11 = [0,
                     0.700802417602134,
                     -0.0940917003303417,
                     0.00476516175202668,
                     -0.000128998651066594,
                     2.18499864260814e-06,
                     -2.53938254792772e-08,
                     2.15276163023997e-10,
                     -1.39082223782012e-12,
                     7.07736209549220e-15]
        self.coef21 = [0,
                       0.700802417602134,
                       -0.188183400660683,
                       0.0142954852560800,
                       -0.000515994604266375,
                       1.09249932130407e-05,
                       -1.52362952875663e-07,
                       1.50693314116798e-09,
                       -1.11265779025609e-11,
                       6.36962588594298e-14]
        self.coef12 = [0.892479070097153,
                       -0.634418280724547,
                       0.0668213136994578,
                       -0.00325082263430659,
                       9.02475797129487e-05,
                       -1.61448458844806e-06,
                       2.01332109031048e-08,
                       -1.84883101478958e-10,
                       1.30180750716843e-12,
                       -7.24995760563704e-15]
        self.coef22 = [0,
                       -0.634418280724547,
                       0.133642627398916,
                       -0.00975246790291976,
                       0.000360990318851795,
                       -8.07242294224032e-06,
                       1.20799265418629e-07,
                       -1.29418171035271e-09,
                       1.04144600573475e-11,
                       -6.52496184507333e-14]
        self.true_eigen = 0.623365592493772
        
    def f_tf(self, x, y, z):
        return -tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y

    def true_y(self, x):
        shapex = tf.shape(x)
        phi1 = 0 * x[:,0]
        phi2 = 0 * x[:,1]
        for m in range(self.N):
            phi1 = phi1 + tf.sin(m * x[:,0]) * self.coef11[m] 
            phi2 = phi2 + tf.cos(m * x[:,0]) * self.coef12[m]
        temp = phi1 * phi2
        return tf.reshape(temp,[shapex[0],1])
    
    def true_z(self, x):
        phi1 = 0 * x[:,0]
        phi2 = 0 * x[:,1]
        dphi1 = 0 * x
        dphi2 = 0 * x
        for m in range(self.N):
            phi1 = phi1 + tf.sin(m * x[:,0]) * self.coef11[m] 
            phi2 = phi2 + tf.cos(m * x[:,1]) * self.coef12[m]
            dphi1 = dphi1 + tf.cos(m * x[:,0]) * self.coef21[m]
            dphi2 = dphi2 + tf.sin(m * x[:,1]) * self.coef22[m]
        return phi1 * dphi2 + phi2 * dphi1
    
    
class Schrodinger6Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d
    # same as Schrodinger3, but we use invariant measure to sample
    def __init__(self, eqn_config):
        super(Schrodinger6Eigen, self).__init__(eqn_config)
        self.h = 0.001;
        self.N = 10
        self.sigma = np.sqrt(2.0)
        self.ci = [0.814723686393179,0.905791937075619,0.126986816293506,\
                   0.913375856139019,0.632359246225410]
        self.sup = [1.56400342750522,1.59706136953917,1.12365661150691,1.59964582497994,1.48429801251911]
        self.coef = [[0.904929598872363, 0.892479070097153, 0.996047011600309, 0.891473839010153, 0.931658124581625],
                     [-0.599085866194182, -0.634418280724547, -0.125605497450144, -0.637155464900669, -0.512357488495517],
                     [0.0573984887007386, 0.0668213136994577, 0.00199001952859007, 0.0676069817707047, 0.0389137499036419],
                     [-0.00252519085534048, -0.00325082263430658, -1.40271491734589e-05, -0.00331506767448259, -0.00134207463713329],
                     [6.32514967687960e-05, 9.02475797129485e-05, 5.56371876669565e-08, 9.27770452504068e-05, 2.62423654134247e-05],
                     [-1.01983519526066e-06, -1.61448458844806e-06, -1.41256489759314e-10, -1.67334312048648e-06, -3.29635802399548e-07],
                     [1.14553116486126e-08, 2.01332109031047e-08, 2.49070180873992e-13, 2.10393672744256e-08, 2.88136027661028e-09],
                     [-9.47170798515556e-11, -1.84883101478958e-10, -3.22670848516871e-16, -1.94804537710462e-10, -1.85271772034910e-11],
                     [6.00357997713989e-13, 1.30180750716843e-12, 3.20055741153896e-19, 1.38305603550407e-12, 9.12827390668906e-14],
                     [-3.00925873281827e-15, -7.24995760563703e-15, -2.50838986315946e-22, -7.76650738246465e-15, -3.55551904006216e-16]]
        self.coef2 = [[0, 0, 0, 0, 0],
                      [-0.599085866194182, -0.634418280724547, -0.125605497450144, -0.637155464900669, -0.512357488495517],
                      [0.114796977401477, 0.133642627398915, 0.00398003905718015, 0.135213963541409, 0.0778274998072837],
                      [-0.00757557256602144, -0.00975246790291974, -4.20814475203766e-05, -0.00994520302344776, -0.00402622391139988],
                      [0.000253005987075184, 0.000360990318851794, 2.22548750667826e-07, 0.000371108181001627, 0.000104969461653699],
                      [-5.09917597630331e-06, -8.07242294224030e-06, -7.06282448796569e-10, -8.36671560243239e-06, -1.64817901199774e-06],
                      [6.87318698916753e-08, 1.20799265418628e-07, 1.49442108524395e-12, 1.26236203646554e-07, 1.72881616596617e-08],
                      [-6.63019558960889e-10, -1.29418171035270e-09, -2.25869593961810e-15, -1.36363176397323e-09, -1.29690240424437e-10],
                      [4.80286398171191e-12, 1.04144600573474e-11, 2.56044592923117e-18, 1.10644482840325e-11, 7.30261912535125e-13],
                      [-2.70833285953644e-14, -6.52496184507332e-14, -2.25755087684351e-21, -6.98985664421818e-14, -3.19996713605594e-15]]
        self.true_eigen = -1.099916247175464
    
    def GradientLnPhi_tf(self, x):
        return 2 * self.true_z(x)/self.sigma / self.true_y(x)
    
    def GradientLnPhi_np(self, x):
        # Phi is square of self.true_y_np
        return 2 * self.true_z_np(x)/self.sigma / self.true_y_np(x)
 

    def sample_general_old(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        for j in range(self.dim):
            MCsample = []
            while len(MCsample) < num_sample:
                # each time add num_sample samples
                x_smp = np.random.uniform(0.0, 2*np.pi, size=[num_sample])
                #pdf = 0.5 #value of function f(x_smp)
                bases_cos = 0 * x_smp
                for m in range(self.N):
                    bases_cos = bases_cos + np.cos(m * x_smp) * self.coef[m][j]
                pdf = np.square(bases_cos)
                reject = np.random.uniform(0.0, self.sup[j], size=[num_sample])
                x_smp = x_smp * (np.sign(pdf - reject) + 1) * 0.5
                temp = x_smp[x_smp != 0]
                MCsample.extend(temp)
            x_sample[:, j, 0] = MCsample[0:num_sample]
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.delta_t*\
            self.GradientLnPhi_np(x_sample[:, :, i]) + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample
    
    def f_tf(self, x, y, z):
        temp = self.GradientLnPhi_tf(x)
        return -tf.reduce_sum(temp * z / self.sigma, axis=1, keepdims=True) - tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y

    def true_y_np(self, x):
        # x in shape [num_sample, dim]
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m]  # Broadcasting
        return np.prod(bases_cos, axis=1, keepdims=True)

    def true_z_np(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + np.sin(m * x) * self.coef2[m] #Broadcasting
        y = np.prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma
    
    def true_y(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def true_z(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + tf.sin(m * x) * self.coef2[m] #Broadcasting
        y = tf.reduce_prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma


class Schrodinger7Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d
    # same as Schrodinger, but we use invariant measure to sample
    def __init__(self, eqn_config):
        super(Schrodinger7Eigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.ci = [0.814723686393179,0.905791937075619]
        self.N = 10
        self.sup = [1.56400342750522 ** 2,1.59706136953917 ** 2]
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
        self.true_eigen = -0.591624518674115
        
    def GradientLnPhi_tf(self, x):
        return 2 * self.true_z(x)/self.sigma / self.true_y(x)
    
    def GradientLnPhi_np(self, x):
        # Phi is square of self.true_y_np
        return 2 * self.true_z_np(x)/self.sigma / self.true_y_np(x)

    def sample_general_old(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        for j in range(self.dim):
            MCsample = []
            while len(MCsample) < num_sample:
                x_smp = np.random.uniform(0.0, 2*np.pi, size=[num_sample])
                bases_cos = 0 * x_smp
                for m in range(self.N):
                    bases_cos = bases_cos + np.cos(m * x_smp) * self.coef[m][j]
                pdf = np.square(bases_cos)
                reject = np.random.uniform(0.0, self.sup[j], size=[num_sample])
                x_smp = x_smp * (np.sign(pdf - reject) + 1) * 0.5
                temp = x_smp[x_smp != 0]
                MCsample.extend(temp)
            x_sample[:, j, 0] = MCsample[0:num_sample]
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.delta_t*\
            self.GradientLnPhi_np(x_sample[:, :, i]) + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample
    
    def f_tf(self, x, y, z):
        temp = self.GradientLnPhi_tf(x)
        return -tf.reduce_sum(temp * z / self.sigma, axis=1, keepdims=True) - tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y

    def true_y_np(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m]  # Broadcasting
        return np.prod(bases_cos, axis=1, keepdims=True)

    def true_z_np(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + np.sin(m * x) * self.coef2[m] #Broadcasting
        y = np.prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma
    
    def true_y(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def true_z(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + tf.sin(m * x) * self.coef2[m] #Broadcasting
        y = tf.reduce_prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma


class Schrodinger8Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d
    # same as Schrodinger, but we use invariant measure to sample
    def __init__(self, eqn_config):
        super(Schrodinger8Eigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.N = 10
        self.ci = [0.814723686393179,0.905791937075619,0.126986816293506,\
                   0.913375856139019,0.632359246225410,0.097540404999410,\
                   0.278498218867048,0.546881519204984,0.957506835434298,0.964888535199277]
        # supremum of each dimension's eigenfunction's absolute value squared
        self.sup = [1.56400342750522 ** 2,1.59706136953917 ** 2,1.12365661150691 ** 2,1.59964582497994 ** 2,1.48429801251911 ** 2,1.09574523792265 ** 2,1.25645243657419 ** 2,1.43922742759836 ** 2,1.61421652060220 ** 2,1.61657863431884 ** 2]
        self.coef = [[0.904929598872363, 0.892479070097153, 0.996047011600309, 0.891473839010153, 0.931658124581625, 0.997649023058051, 0.982280529025825, 0.944649556846450, 0.885723649385784, 0.884778462161165],
                     [-0.599085866194182, -0.634418280724547, -0.125605497450144, -0.637155464900669, -0.512357488495517, -0.0969095482431325, -0.264889328009242, -0.462960196131467, -0.652506034959353, -0.654980719478657],
                     [0.0573984887007386, 0.0668213136994577, 0.00199001952859007, 0.0676069817707047, 0.0389137499036419, 0.00118025466788750, 0.00914049555158614, 0.0306829183857062, 0.0721762094875563, 0.0729397325313142],
                     [-0.00252519085534048, -0.00325082263430658, -1.40271491734589e-05, -0.00331506767448259, -0.00134207463713329, -6.39243615673157e-06, -0.000140854107058429, -0.000919007154831146, -0.00370016361849726, -0.00376642561780699],
                     [6.32514967687960e-05, 9.02475797129485e-05, 5.56371876669565e-08, 9.27770452504068e-05, 2.62423654134247e-05, 1.94793735338997e-08, 1.22305183605055e-06, 1.55782949374877e-05, 0.000108388631038444, 0.000111150892928655],
                     [-1.01983519526066e-06, -1.61448458844806e-06, -1.41256489759314e-10, -1.67334312048648e-06, -3.29635802399548e-07, -3.79928584585742e-11, -6.80228425921455e-09, -1.69495102259451e-07, -2.04729080278141e-06, -2.11528784557100e-06],
                     [1.14553116486126e-08, 2.01332109031047e-08, 2.49070180873992e-13, 2.10393672744256e-08, 2.88136027661028e-09, 5.14616943511569e-14, 2.62845162481860e-11, 1.28269245292752e-09, 2.69656139309183e-08, 2.80726379468963e-08],
                     [-9.47170798515556e-11, -1.84883101478958e-10, -3.22670848516871e-16, -1.94804537710462e-10, -1.85271772034910e-11, -5.12132251946685e-17, -7.46405433786103e-14, -7.13859314209407e-12, -2.61601555334212e-10, -2.74416243017770e-10],
                     [6.00357997713989e-13, 1.30180750716843e-12, 3.20055741153896e-19, 1.38305603550407e-12, 9.12827390668906e-14, 3.90213598890932e-20, 1.62311055314169e-16, 3.04361603796089e-14, 1.94624368648372e-12, 2.05717944603459e-12],
                     [-3.00925873281827e-15, -7.24995760563703e-15, -2.50838986315946e-22, -7.76650738246465e-15, -3.55551904006216e-16, -2.34921193685406e-23, -2.78916493656665e-19, -1.02576252116579e-16, -1.14534279420053e-14, -1.21989378354622e-14]]

        self.coef2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [-0.599085866194182, -0.634418280724547, -0.125605497450144, -0.637155464900669, -0.512357488495517, -0.0969095482431325, -0.264889328009242, -0.462960196131467, -0.652506034959353, -0.654980719478657],
                      [0.114796977401477, 0.133642627398915, 0.00398003905718015, 0.135213963541409, 0.0778274998072837, 0.00236050933577499, 0.0182809911031723, 0.0613658367714124, 0.144352418975113, 0.145879465062628],
                      [-0.00757557256602144, -0.00975246790291974, -4.20814475203766e-05, -0.00994520302344776, -0.00402622391139988, -1.91773084701947e-05, -0.000422562321175286, -0.00275702146449344, -0.0111004908554918, -0.0112992768534210],
                      [0.000253005987075184, 0.000360990318851794, 2.22548750667826e-07, 0.000371108181001627, 0.000104969461653699, 7.79174941355989e-08, 4.89220734420220e-06, 6.23131797499510e-05, 0.000433554524153778, 0.000444603571714619],
                      [-5.09917597630331e-06, -8.07242294224030e-06, -7.06282448796569e-10, -8.36671560243239e-06, -1.64817901199774e-06, -1.89964292292871e-10, -3.40114212960727e-08, -8.47475511297256e-07, -1.02364540139071e-05, -1.05764392278550e-05],
                      [6.87318698916753e-08, 1.20799265418628e-07, 1.49442108524395e-12, 1.26236203646554e-07, 1.72881616596617e-08, 3.08770166106942e-13, 1.57707097489116e-10, 7.69615471756512e-09, 1.61793683585510e-07, 1.68435827681378e-07],
                      [-6.63019558960889e-10, -1.29418171035270e-09, -2.25869593961810e-15, -1.36363176397323e-09, -1.29690240424437e-10, -3.58492576362679e-16, -5.22483803650272e-13, -4.99701519946585e-11, -1.83121088733948e-09, -1.92091370112439e-09],
                      [4.80286398171191e-12, 1.04144600573474e-11, 2.56044592923117e-18, 1.10644482840325e-11, 7.30261912535125e-13, 3.12170879112745e-19, 1.29848844251335e-15, 2.43489283036871e-13, 1.55699494918698e-11, 1.64574355682767e-11],
                      [-2.70833285953644e-14, -6.52496184507332e-14, -2.25755087684351e-21, -6.98985664421818e-14, -3.19996713605594e-15, -2.11429074316866e-22, -2.51024844290998e-18, -9.23186269049210e-16, -1.03080851478047e-13, -1.09790440519160e-13]]
        self.true_eigen = -1.986050602989757
        
    def GradientLnPhi_tf(self, x):
        return 2 * self.true_z(x)/self.sigma / self.true_y(x)
    
    def GradientLnPhi_np(self, x):
        # Phi is square of self.true_y_np
        return 2 * self.true_z_np(x)/self.sigma / self.true_y_np(x)
 

    def sample_general_old(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        for j in range(self.dim):
            MCsample = []
            while len(MCsample) < num_sample:
                x_smp = np.random.uniform(0.0, 2*np.pi, size=[num_sample])
                bases_cos = 0 * x_smp
                for m in range(self.N):
                    bases_cos = bases_cos + np.cos(m * x_smp) * self.coef[m][j]
                pdf = np.square(bases_cos)
                reject = np.random.uniform(0.0, self.sup[j], size=[num_sample])
                x_smp = x_smp * (np.sign(pdf - reject) + 1) * 0.5
                temp = x_smp[x_smp != 0]
                MCsample.extend(temp)
            x_sample[:, j, 0] = MCsample[0:num_sample]
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.delta_t*\
            self.GradientLnPhi_np(x_sample[:, :, i]) + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample
    
    def f_tf(self, x, y, z):
        temp = self.GradientLnPhi_tf(x)
        return -tf.reduce_sum(temp * z / self.sigma, axis=1, keepdims=True) - tf.reduce_sum(self.ci * tf.cos(x), axis=1, keepdims=True) *y

    def true_y_np(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m]  # Broadcasting
        return np.prod(bases_cos, axis=1, keepdims=True)

    def true_z_np(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + np.sin(m * x) * self.coef2[m] #Broadcasting
        y = np.prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma
    
    def true_y(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def true_z(self, x):
        bases_cos = 0
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(m * x) * self.coef[m] #Broadcasting
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + tf.sin(m * x) * self.coef2[m] #Broadcasting
        y = tf.reduce_prod(bases_cos, axis=1, keepdims=True)
        return - y * bases_sin / bases_cos * self.sigma


class NonlinearEigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d
    # same as Schrodinger, but we use invariant measure to sample
    def __init__(self, eqn_config):
        super(NonlinearEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.true_eigen = 0.0
        
    def f_tf(self, x, y, z):
        temp = tf.reduce_sum(tf.square(z/self.sigma),axis=1,keepdims=True)
        return tf.reduce_sum(tf.cos(x), axis=1, keepdims=True) * y - temp / y
        
    def true_z(self, x):
        temp = tf.exp(tf.reduce_sum(tf.cos(x), axis=1, keepdims=True))
        return - tf.sin(x) * temp * self.sigma #broadcasting

    def true_y(self, x):
        return tf.exp(tf.reduce_sum(tf.cos(x), axis=1, keepdims=True))
    

class CubicSchrodingerEigen(Equation):
    # Cubic Schrodinger L psi = -Delta psi + epsl psi^3 + V psi  dim=2
    # where V(x)= \sum_{i=1}^d (sin^2(xi) - cos(xi)) - epsl* exp(2 \sum_{i=1}^d cos(xi))/4^d -3 
    # on squares [0, 2pi]^d. True eigenvalue=-3, eigenfunction exp(\sum_{i=1}^d cos(xi)) / 2^d
    # \int_{0}^{2pi} exp(2cos(x))dx = 14.32305687810051 = 2pi * 2.279585302336067 
    def __init__(self, eqn_config):
        super(CubicSchrodingerEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.true_eigen = -3.0
        self.epsl = 1
        self.dim = eqn_config.dim #2
        # norm_constant makes true_y has unit L2 mean
        self.norm_constant = 1.509829560690897 #sqrt(2.279585302336067)
        # L2mean is the L2 mean of eigenfunction
        self.L2mean = 0.5
        
    def f_tf(self, x, y, z):
        temp = self.epsl / (2.279585302336067 ** self.dim) * (self.L2mean ** 2) * tf.exp(2 * tf.reduce_sum(tf.cos(x),axis=1,keepdims=True))\
        - tf.reduce_sum(tf.square(tf.sin(x)) - tf.cos(x), axis=1, keepdims=True)
        #return -self.epsl * tf.pow(y,3) + (temp + 3.0) * self.true_y(x)
        return -self.epsl * tf.pow(y,3) + (temp + 3.0) * y
        
    def true_z(self, x):
        temp = tf.exp(tf.reduce_sum(tf.cos(x), axis=1, keepdims=True))
        return - tf.sin(x) * temp * self.sigma /(1.509829560690897 ** self.dim) * self.L2mean #broadcasting

    def true_y(self, x):
        return tf.exp(tf.reduce_sum(tf.cos(x), axis=1, keepdims=True)) / (1.509829560690897 ** self.dim) * self.L2mean

class CubicSchrodinger2Eigen(Equation):
    # Cubic Schrodinger L psi = -Delta psi + epsl psi^3 + V psi   dim=5
    # where V(x)= \sum_{i=1}^d (sin^2(xi) - cos(xi)) - epsl* exp(2 \sum_{i=1}^d cos(xi))/4^d -3 
    # on squares [0, 2pi]^d. True eigenvalue=-3, eigenfunction exp(\sum_{i=1}^d cos(xi)) / 2^d
    # \int_{0}^{2pi} exp(2cos(x))dx = 14.3231 = 2pi * 2.27959 
    def __init__(self, eqn_config):
        super(CubicSchrodinger2Eigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.true_eigen = -3.0
        self.epsl = eqn_config.epsl
        # L2mean is the L2 mean of eigenfunction
        self.L2mean = eqn_config.L2mean
        self.dim = eqn_config.dim #5
        # norm_constant makes true_y has unit L2 mean
        self.norm_constant = 1.509829560690897 #sqrt(2.27959)

    def f_tf(self, x, y, z):
        temp = self.epsl / (2.279585302336067 ** self.dim) * (self.L2mean ** 2) * tf.exp(2 * tf.reduce_sum(tf.cos(x),axis=1,keepdims=True))\
        - tf.reduce_sum(tf.square(tf.sin(x)) - tf.cos(x), axis=1, keepdims=True)
        #return -self.epsl * tf.pow(y,3) + (temp + 3.0) * self.true_y(x)
        return -self.epsl * tf.pow(y,3) + (temp + 3.0) * y
        
    def true_z(self, x):
        temp = tf.exp(tf.reduce_sum(tf.cos(x), axis=1, keepdims=True))
        return - tf.sin(x) * temp * self.sigma /(self.norm_constant ** self.dim) * self.L2mean #broadcasting

    def true_y(self, x):
        return tf.exp(tf.reduce_sum(tf.cos(x), axis=1, keepdims=True)) / (self.norm_constant ** self.dim) * self.L2mean


class CubicNewEigen(Equation):
    def __init__(self, eqn_config):
        super(CubicNewEigen, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.true_eigen = -3.0
        self.epsl = eqn_config.epsl
        self.dim = eqn_config.dim
        # norm_constant makes true_y has unit L2 mean
        # sqrt((integral exp(2*cos(x)/dim) from 0 to 2*pi) / 2/pi) ** dim
        self.norm_constant = eqn_config.norm_constant
        # L2mean is the L2 mean of eigenfunction
        self.L2mean = eqn_config.L2mean

    def f_tf(self, x, y, z):
        temp = self.epsl / (self.norm_constant**2) * (self.L2mean ** 2) * tf.exp(2 * tf.reduce_mean(tf.cos(x), axis=1, keepdims=True)) \
               - tf.reduce_sum(tf.square(tf.sin(x)/self.dim) - tf.cos(x)/self.dim, axis=1, keepdims=True)
        # return -self.epsl * tf.pow(y,3) + (temp + 3.0) * self.true_y(x)
        return -self.epsl * tf.pow(y, 3) + (temp + 3.0) * y

    def true_z(self, x):
        temp = tf.exp(tf.reduce_mean(tf.cos(x), axis=1, keepdims=True))
        return - tf.sin(x) / self.dim * temp * self.sigma / self.norm_constant * self.L2mean  # broadcasting

    def true_y(self, x):
        return tf.exp(tf.reduce_mean(tf.cos(x), axis=1, keepdims=True)) / self.norm_constant * self.L2mean