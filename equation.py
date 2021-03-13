import numpy as np
from scipy.stats import multivariate_normal as normal

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf

class Equation(object):
    """Base class for defining PDE related function."""
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = (self.total_time + 0.0) / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.sigma = None
        
    def sample_hist(self, num_sample):
        x_sample = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        return x_sample

    def sample_uniform(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        if self.dim == 1:
            dw_sample = np.reshape(dw_sample,[num_sample, 1, self.num_time_interval] )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        # uniform X0
        x_sample[:, :, 0] = np.random.uniform(0.0, 2*np.pi, size=[num_sample, self.dim])
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class FPEigen(Equation):
    # eigenvalue problem for Fokker Planck operator on squares [0, 2pi]^d
    # uniform sampling with no drift term
    # v(x) = sin(\sum_{i=1}^d c_i cos(x_i))   psi = exp(-v)
    def __init__(self, eqn_config):
        super(FPEigen, self).__init__(eqn_config)
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


class Sdg_d10Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d ci~[0,0.2]
    def __init__(self, eqn_config):
        super(Sdg_d10Eigen, self).__init__(eqn_config)
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


class Sdg_d5Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d ci~[0,0.2]
    def __init__(self, eqn_config):
        super(Sdg_d5Eigen, self).__init__(eqn_config)
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


class Sdg_d2Eigen(Equation):
    # Schrodinger V(x)= \sum_{i=1}^d ci*cos(xi) on squares [0, 2pi]^d ci~[0,0.2]
    def __init__(self, eqn_config):
        super(Sdg_d2Eigen, self).__init__(eqn_config)
        self.N = 10
        self.sigma = np.sqrt(2.0)
        self.ci = [0.814723686393179,0.905791937075619]
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
    
    
class CubicSdgEigen(Equation):
    def __init__(self, eqn_config):
        super(CubicSdgEigen, self).__init__(eqn_config)
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
    

class DWClose_d1Eigen(Equation):
    #degenerate Schrodinger V(x)= A cos(2x) on [0, 2pi]
    def __init__(self, eqn_config):
        super(DWClose_d1Eigen, self).__init__(eqn_config)
        self.N = 10
        self.sigma = np.sqrt(2.0)
        self.A = 5
        self.true_eigen = -2.15307834
        self.second_eigen = -2.07633151
        # coef0 is coef for ground eigenfunction: 1, cos2x, cos4x...
        self.coef0 = [4.259287663084366793e-01,-3.668232007964972730e-01,
                      5.097922425487490944e-02,-3.348739889607468302e-03,
                      1.266698872480900102e-04,-3.101300263994905906e-06,
                      5.306028830867799572e-08,-6.695172840333706772e-10,
                      6.484195313839146743e-12,-4.970641756370646460e-14]
        # coef1 is coef for second eigenfunction: sinx, sin3x, sin5x...
        self.coef1 = [6.888844529501658709e-01,-1.588103256443261224e-01,
                      1.472987240390475835e-02,-7.220376543484285491e-04,
                      2.174143337189041368e-05,-4.417561381727552544e-07,
                      6.456580422456261813e-09,-7.109053841528895779e-11,
                      6.106194946251028526e-13,-4.204648637005190987e-15]

    def f_tf(self, x, y, z):
        return -self.A * tf.cos(2*x) *y
       
    def true_y_np(self,x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + np.cos(2 * m * x) * self.coef0[m]
        return bases_cos
        #return np.prod(bases_cos, axis=1, keepdims=True)
    
    def true_y(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(2 * m * x) * self.coef0[m]
        return bases_cos
        #return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def second_y(self, x):
        bases_sin = 0 * x
        for m in range(self.N):
            bases_sin = bases_sin + tf.sin((2*m+1) * x) * self.coef1[m]
        return bases_sin
    
    def second_y_approx(self, x):
        # you can choose whatever function you like as initialization
        # bases_sin = 0 * x
        # for m in range(1):
        #     bases_sin = bases_sin + tf.sin((2*m+1) * x) * self.coef1[m]
        return 0.3*self.true_y(x) + 3*self.second_y(x)
    
    def true_z(self, x):
        bases_sin = 0
        for m in range(self.N):
            bases_sin = bases_sin + 2 * m * tf.sin(2 * m * x) * self.coef0[m]
        return - bases_sin * self.sigma
    
    def second_z(self, x):
        bases_cos = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + (2*m+1) * tf.cos((2*m+1) *x) * self.coef1[m]
        return bases_cos * self.sigma
    
    def second_z_approx(self, x):
        # set as the gradient of second_y_approx
        # bases_cos = 0 * x
        # for m in range(1):
        #     bases_cos = bases_cos + (2*m+1) * tf.cos((2*m+1) *x) * self.coef1[m]
        return 0.3*self.true_z(x) + 3*self.second_z(x)

    
class DWSepEigen(Equation):
    # well-separate Schrodinger example for arbitrary dimensions
    def __init__(self, eqn_config):
        super(DWSepEigen, self).__init__(eqn_config)
        self.N = 10
        self.sigma = np.sqrt(2.0)
        self.A = np.concatenate(([1.5],0.2*np.ones(self.dim-1)))
        self.true_eigen = -0.2658780338622622197 - 0.004994543800531658917 * (self.dim-1)
        self.second_eigen = 0.1860163214034415979 - 0.004994543800531658917 * (self.dim-1)
        # coef0 is coef for ground eigenfunction in d1: 1, cos2x, cos4x...
        self.coef0 = [4.849638936137847245e-01,-1.719216620376243232e-01,
                      7.934668280954022823e-03,-1.641332919106110351e-04,
                      1.915646686059286880e-06,-1.432980906804100879e-08,
                      7.449835308605464979e-11,-2.846866874407665454e-13,
                      8.247577849078903519e-16,-1.840772951476521254e-18]
        # coef1 is coef for second eigenfunction in d1: sinx, sin3x, sin5x...  
        self.coef1 = [7.045452301176932108e-01,-6.010586074745370710e-02,
                      1.817537370379718794e-03,-2.792944433460181340e-05,
                      2.592161556682200467e-07,-1.609229933303384171e-09,
                      7.149528211141171171e-12,-2.385168629168197918e-14,
                      6.193904135450362291e-17,-1.287490135293999248e-19]
        # coef2 is coef for ground eigenfunction in other dim: 1, cos2x, cos4x...
        self.coef2 = [4.996884608914199388e-01,-2.495715904542338340e-02,
                      1.559362739878917203e-04,-4.330981093820056683e-07,
                      6.766640464567428604e-10,-6.766307217324653586e-13,
                      4.698663151849139795e-16,-2.397216509273572312e-19,
                      9.363952505944370821e-23,3.060820295240397227e-26]
        # coef3 is to put coef0 and coef2s together, this is not a list
        self.coef3 = np.concatenate([
            np.reshape(self.coef0,[self.N,1]), 
            np.repeat(np.reshape(self.coef2,[self.N,1]), self.dim-1, axis=1)
            ],axis=1
        )
    
    def f_tf(self, x, y, z):
        return - y * tf.reduce_sum(self.A * tf.cos(2*x), axis=1, keepdims=True)
    
    def true_y(self, x):
        bases_cos = x*0
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(2*m*x) * self.coef3[m,:] #broadcast
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True)
    
    def second_y(self, x):
        bases_cos = x[:,1:self.dim]*0
        bases_sin = x[:,0:1]*0
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(2*m*x[:,1:self.dim]) * self.coef3[m,1:self.dim] #broadcast
            bases_sin = bases_sin + tf.sin((2*m+1) * x[:,0:1]) * self.coef1[m]
        return tf.reduce_prod(bases_cos, axis=1, keepdims=True) * bases_sin

    def second_y_approx(self, x):
        # you can choose whatever function you like as initialization
        return 0.3*self.true_y(x) + 3*self.second_y(x)
    
    def true_z(self, x):
        bases_cos = 0 * x
        bases_sin = 0 * x
        for m in range(self.N):
            bases_cos = bases_cos + tf.cos(2*m*x) * self.coef3[m,:]
            bases_sin = bases_sin - 2 * m * tf.sin(2*m*x) * self.coef3[m,:]
        y = tf.reduce_prod(bases_cos, axis=1, keepdims=True)
        return self.sigma * y * bases_sin / bases_cos
    
    def second_z(self, x):
        y1 = 0 * x[:,0:1]
        y2 = 0 * x[:,1:self.dim]
        dy1 = 0 * x[:,0:1]
        dy2 = 0 * x[:,1:self.dim]
        for m in range(self.N):
            y1 = y1 + tf.sin((2*m+1) * x[:,0:1]) * self.coef1[m]
            y2 = y2 + tf.cos(2 * m * x[:,1:self.dim]) * self.coef3[m,1:self.dim]
            dy1 = dy1 + (2*m+1) * tf.cos((2*m+1) * x[:,0:1]) * self.coef1[m]
            dy2 = dy2 - 2*m * tf.sin(2*m * x[:,1:self.dim]) * self.coef3[m,1:self.dim]
        y = tf.reduce_prod(y2, axis=1, keepdims=True) * y1
        yy = tf.concat([y1,y2],1) # expand y
        zz = tf.concat([dy1,dy2],1) # expand z
        return y * zz / yy
    
    def second_z_approx(self, x):
        # set as the gradient of second_y_approx
        return 0.3*self.true_z(x) + 3*self.second_z(x)
