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
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        MCsample = np.zeros(shape=[0, self.dim])
        sup = 1
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
    
    def f_tf(self, x, y, z):
        return -y * tf.cos(x[:,0:1])
#        return y * self.laplician_v(self, x)

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
                #pdf = 0.5 #value of function f(x_smp)
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
        N = 10
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        #x_sample[:, :, 0] = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        for j in range(self.dim):
            MCsample = []
            while len(MCsample) < num_sample:
                # each time add num_sample samples
                x_smp = np.random.uniform(0.0, 2*np.pi, size=[num_sample])
                bases_cos = 0 * x_smp
                for m in range(N):
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