import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

TF_DTYPE = tf.float64
DELTA_CLIP = 100.0


class FeedForwardModel(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde, sess):
        self.eqn_config = config.eqn_config
        self.nn_config = config.nn_config
        self.bsde = bsde
        self.sess = sess
        self.y_init = None
        # make sure consistent with FBSDE equation
        self.dim = bsde.dim
        self.num_time_interval = bsde.num_time_interval
        self.total_time = bsde.total_time
        # ops for statistics update of batch normalization
        self.extra_train_ops = []
        self.dw = tf.placeholder(TF_DTYPE, [None, self.dim, self.num_time_interval], name='dW')
        self.x = tf.placeholder(TF_DTYPE, [None, self.dim, self.num_time_interval + 1], name='X')
        self.train_loss, self.eigen_error, self.init_rel_loss, self.NN_consist,self.l2 = None, None, None, None, None
        self.train_ops, self.t_build = None, None
        self.eigen = tf.get_variable('eigen', shape=[1], dtype=TF_DTYPE,
                                     initializer=tf.random_uniform_initializer(self.eqn_config.initeigen_low, self.eqn_config.initeigen_high),
                                     trainable=True)
        self.x_hist = tf.placeholder(TF_DTYPE, [None, self.dim], name='x_hist')
        self.hist_NN = None
        self.hist_true = None
        self.hist_size = 20000
    
    def train(self):
        start_time = time.time()
        # to save iteration results
        training_history = []
        # for validation
        dw_valid, x_valid = self.bsde.sample_uniform(self.nn_config.valid_size)
        #dw_valid, x_valid = self.bsde.sample(self.nn_config.valid_size)
        # can still use batch norm of samples in the validation phase
        feed_dict_valid = {self.dw: dw_valid, self.x: x_valid}
        # initialization
        self.sess.run(tf.global_variables_initializer())
        # begin sgd iteration
        for step in range(self.nn_config.num_iterations+1):
            if step % self.nn_config.logging_frequency == 0:
                train_loss, eigen_error, init_rel_loss, grad_error, NN_consist,l2 = self.sess.run(
                    [self.train_loss, self.eigen_error, self.init_rel_loss, self.grad_error, self.NN_consist, self.l2],
                    feed_dict=feed_dict_valid)
                elapsed_time = time.time()-start_time+self.t_build
                training_history.append([step, train_loss, eigen_error, init_rel_loss, grad_error,NN_consist, l2, elapsed_time])
                if self.nn_config.verbose:
                    logging.info(
                        "step: %5u,    train_loss: %.4e,   eigen_error: %.4e, grad_error: %.4e, NN_consist: %.4e, l2: %.4e " % (
                            step, train_loss, eigen_error, grad_error, NN_consist, l2) +
                        "init_rel_loss: %.4e,   elapsed time %3u" % (
                         init_rel_loss, elapsed_time))
            dw_train, x_train = self.bsde.sample_uniform(self.nn_config.batch_size)
            #dw_train, x_train = self.bsde.sample(self.nn_config.batch_size)
            self.sess.run(self.train_ops, feed_dict={self.dw: dw_train, self.x: x_train})
            if step == self.nn_config.num_iterations:
                x_hist = self.bsde.sample_hist(self.hist_size)
                feed_dict_hist = {self.x_hist: x_hist}
                [y_hist, y_true] = self.sess.run([self.hist_NN, self.hist_true], feed_dict=feed_dict_hist)
        return np.array(training_history), y_hist, y_true

    def build_linear_consist(self):
        start_time = time.time()
        with tf.variable_scope('forward'):
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False, dtype=tf.int32)
            decay = tf.train.piecewise_constant(
                global_step, self.nn_config.ma_boundaries,
                [tf.constant(ma, dtype=TF_DTYPE) for ma in self.nn_config.ma_values])
            x_init = self.x[:, :, 0]
            net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            net_z = PeriodNet(self.nn_config.num_hiddens, out_dim=self.dim, name='net_z')
            y_init_and_gradient = net_y(x_init,need_grad=True)
            y_init = y_init_and_gradient[0]
            grad_y = y_init_and_gradient[1]
            z = net_z(x_init, need_grad=False)
            z_init = z
            
            yl2_batch = tf.reduce_mean(y_init ** 2)
            yl2_ma = tf.get_variable(
                'yl2_ma', [1], TF_DTYPE,
                initializer=tf.constant_initializer(100.0, TF_DTYPE),
                trainable=False)
            yl2 = decay * yl2_ma + (1 - decay) * yl2_batch
            true_z = self.bsde.true_z(x_init)
            
            sign = tf.sign(tf.reduce_sum(y_init))
            normed_true_z = true_z / tf.sqrt(tf.reduce_mean(true_z ** 2))
            error_z = z / tf.sqrt(tf.reduce_mean(z ** 2)) - normed_true_z
            y_init = y_init / tf.sqrt(yl2) * sign
            grad_y = grad_y * sign / tf.sqrt(yl2)
            NN_consist_0 = z_init - grad_y
            
            x_T = self.x[:, :, -1]
            z_T = net_z(x_T, need_grad=False)
            yT_and_gradient = net_y(x_T,need_grad=True)
            grad_yT = yT_and_gradient[1]
            grad_yT = grad_yT * sign / tf.sqrt(yl2)
            NN_consist_T = z_T - grad_yT
            
            #NN_consist = NN_consist_0
            NN_consist = NN_consist_T
            
            y = y_init
            
            for t in range(0, self.num_time_interval-1):
                y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen *y) + \
                    tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                z = net_z(self.x[:, :, t + 1], need_grad=False, reuse=True)
            # terminal time
            y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen *y) + \
                tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)
            y_xT = net_y(self.x[:, :, -1], need_grad=False, reuse=True)
            y_xT = y_xT / tf.sqrt(yl2) * sign
            delta = y - y_xT
            
            # use linear approximation outside the clipped range
            self.train_loss = tf.reduce_mean(
                tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                         2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 1000 \
                    + tf.reduce_mean(tf.square(NN_consist)) * 20\
                    + tf.nn.relu(2 - yl2) * 100
            self.extra_train_ops.append(
                moving_averages.assign_moving_average(yl2_ma, yl2_batch, decay))
            y_hist = net_y(self.x_hist, need_grad=False, reuse=True)
            hist_sign = tf.sign(tf.reduce_sum(y_hist))
            hist_l2 = tf.reduce_mean(y_hist ** 2)
            self.hist_NN = y_hist / tf.sqrt(hist_l2) * hist_sign
            hist_true_y = self.bsde.true_y(self.x_hist)
            self.hist_true = hist_true_y / tf.sqrt(tf.reduce_mean(hist_true_y ** 2))
        true_init = self.bsde.true_y(self.x[:, :, 0])
        true_init = true_init / tf.sqrt(tf.reduce_mean(true_init ** 2))
        error_y = y_init - true_init
        self.init_rel_loss = tf.sqrt(tf.reduce_mean(error_y ** 2))
        self.eigen_error = self.eigen - self.bsde.true_eigen
        self.l2 = yl2
        self.grad_error = tf.sqrt(tf.reduce_mean(error_z ** 2))
        self.NN_consist = tf.sqrt(tf.reduce_mean(NN_consist ** 2))
        
        # train operations
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self.nn_config.lr_boundaries,
                                                    self.nn_config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.train_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')
        all_ops = [apply_op] + self.extra_train_ops
        self.train_ops = tf.group(*all_ops)
        self.t_build = time.time() - start_time
    
    def build_nonlinear_consist(self):
        start_time = time.time()
        with tf.variable_scope('forward'):
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False, dtype=tf.int32)

            decay = tf.train.piecewise_constant(
                global_step, self.nn_config.ma_boundaries,
                [tf.constant(ma, dtype=TF_DTYPE) for ma in self.nn_config.ma_values])
            x_init = self.x[:, :, 0]
            net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            net_z = PeriodNet(self.nn_config.num_hiddens, out_dim=self.dim, name='net_z')
            y_init_and_gradient = net_y(x_init, need_grad=True)
            y_init = y_init_and_gradient[0]
            grad_y = y_init_and_gradient[1]
            z = net_z(x_init, need_grad=False)
            
            yl2_batch = tf.reduce_mean(y_init ** 2)
            yl2_ma = tf.get_variable(
                'yl2_ma', [1], TF_DTYPE,
                initializer=tf.constant_initializer(100.0, TF_DTYPE),
                trainable=False)
            yl2 = decay * yl2_ma + (1 - decay) * yl2_batch
            true_z = self.bsde.true_z(x_init)
            sign = tf.sign(tf.reduce_sum(y_init))
            error_z = z - true_z
            NN_consist_0 = z - grad_y * sign / tf.sqrt(yl2) * self.bsde.L2mean
            
            x_T = self.x[:, :, -1]
            z_T = net_z(x_T, need_grad=False)
            yT_and_gradient = net_y(x_T,need_grad=True)
            grad_yT = yT_and_gradient[1]
            grad_yT = grad_yT * sign / tf.sqrt(yl2) * self.bsde.L2mean
            NN_consist_T = z_T - grad_yT
            
            #NN_consist = NN_consist_0
            NN_consist = NN_consist_T
            
            y = y_init
            y = y * sign / tf.sqrt(yl2) * self.bsde.L2mean
            y = tf.clip_by_value(y, -5, 5, name=None)
            for t in range(0, self.num_time_interval - 1):
                y = y - self.bsde.delta_t * (
                    self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen * y) + \
                    tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                y = tf.clip_by_value(y, -5, 5, name=None)
                z = net_z(self.x[:, :, t + 1], need_grad=False, reuse=True)
            # terminal time
            y = y - self.bsde.delta_t * (
            self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen * y) + \
                tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)

            y_xT = net_y(self.x[:, :, -1], need_grad=False, reuse=True)
            y_xT = y_xT / tf.sqrt(yl2) * sign * self.bsde.L2mean
            delta = y - y_xT
            # use linear approximation outside the clipped range
            self.train_loss = \
                tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                         2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 100 \
                + tf.reduce_mean(tf.square(NN_consist)) * 20\
                + tf.nn.relu(2 * self.bsde.L2mean - yl2) * 100

            self.extra_train_ops.append(
                moving_averages.assign_moving_average(yl2_ma, yl2_batch, decay))
            y_hist = net_y(self.x_hist, need_grad=False, reuse=True)
            hist_sign = tf.sign(tf.reduce_sum(y_hist))
            hist_l2 = tf.reduce_mean(y_hist ** 2)
            self.hist_NN = y_hist / tf.sqrt(hist_l2) * hist_sign
            hist_true_y = self.bsde.true_y(self.x_hist)
            self.hist_true = hist_true_y / tf.sqrt(tf.reduce_mean(hist_true_y ** 2))

        true_init = self.bsde.true_y(self.x[:, :, 0])
        rel_err = tf.reduce_mean(tf.square(
            y_init / tf.sqrt(yl2) * sign * self.bsde.L2mean - true_init)) / tf.reduce_mean(
            tf.square(true_init))
        self.init_rel_loss = tf.sqrt(rel_err)

        self.eigen_error = self.eigen - self.bsde.true_eigen
        self.l2 = yl2
        self.grad_error = tf.sqrt(tf.reduce_mean(error_z ** 2) / tf.reduce_mean(true_z ** 2))
        self.NN_consist = tf.sqrt(tf.reduce_mean(NN_consist ** 2))

        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self.nn_config.lr_boundaries,
                                                    self.nn_config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.train_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')

        all_ops = [apply_op] + self.extra_train_ops
        self.train_ops = tf.group(*all_ops)
        self.t_build = time.time() - start_time
        
    def build_linear_grad(self):
        start_time = time.time()
        with tf.variable_scope('forward'):
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False, dtype=tf.int32)
            decay = tf.train.piecewise_constant(
                global_step, self.nn_config.ma_boundaries,
                [tf.constant(ma, dtype=TF_DTYPE) for ma in self.nn_config.ma_values])
            x_init = self.x[:, :, 0]
            net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            y_init_and_gradient = net_y(x_init,need_grad=True)
            y_init = y_init_and_gradient[0]
            z = y_init_and_gradient[1]
            
            yl2_batch = tf.reduce_mean(y_init ** 2)
            yl2_ma = tf.get_variable(
                'yl2_ma', [1], TF_DTYPE,
                initializer=tf.constant_initializer(100.0, TF_DTYPE),
                trainable=False)
            yl2 = decay * yl2_ma + (1 - decay) * yl2_batch
            
            true_z = self.bsde.true_z(x_init)
            sign = tf.sign(tf.reduce_sum(y_init))
            z = z / tf.sqrt(yl2) * sign
            normed_true_z = true_z / tf.sqrt(tf.reduce_mean(true_z ** 2))
            error_z = z / tf.sqrt(tf.reduce_mean(z ** 2)) - normed_true_z
            y_init = y_init / tf.sqrt(yl2) * sign
            #self.y_init = y_init
            y = y_init
            
            for t in range(0, self.num_time_interval-1):
                y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen *y) + \
                    tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                yz = net_y(self.x[:, :, t + 1], need_grad=True, reuse=True)
                z = yz[1] / tf.sqrt(yl2) * sign
            # terminal time
            y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen *y) + \
                tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)
            y_xT = net_y(self.x[:, :, -1], need_grad=False, reuse=True)
            y_xT = y_xT / tf.sqrt(yl2) * sign
            delta = y - y_xT
            
            # use linear approximation outside the clipped range
            self.train_loss = tf.reduce_mean(
                tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                         2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 1000 \
                    + tf.nn.relu(2 - yl2) * 100
            self.extra_train_ops.append(
                moving_averages.assign_moving_average(yl2_ma, yl2_batch, decay))
            y_hist = net_y(self.x_hist, need_grad=False, reuse=True)
            hist_sign = tf.sign(tf.reduce_sum(y_hist))
            hist_l2 = tf.reduce_mean(y_hist ** 2)
            self.hist_NN = y_hist / tf.sqrt(hist_l2) * hist_sign
            hist_true_y = self.bsde.true_y(self.x_hist)
            self.hist_true = hist_true_y / tf.sqrt(tf.reduce_mean(hist_true_y ** 2))
        true_init = self.bsde.true_y(self.x[:, :, 0])
        true_init = true_init / tf.sqrt(tf.reduce_mean(true_init ** 2))
        error_y = y_init - true_init
        self.init_rel_loss = tf.sqrt(tf.reduce_mean(error_y ** 2))
        self.eigen_error = self.eigen - self.bsde.true_eigen
        self.l2 = yl2
        self.grad_error = tf.sqrt(tf.reduce_mean(error_z ** 2))
        self.NN_consist = tf.constant(0)
        
        # train operations
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self.nn_config.lr_boundaries,
                                                    self.nn_config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.train_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')
        all_ops = [apply_op] + self.extra_train_ops
        self.train_ops = tf.group(*all_ops)
        self.t_build = time.time() - start_time
    
    def build_nonlinear_grad(self):
        start_time = time.time()
        with tf.variable_scope('forward'):
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False, dtype=tf.int32)

            decay = tf.train.piecewise_constant(
                global_step, self.nn_config.ma_boundaries,
                [tf.constant(ma, dtype=TF_DTYPE) for ma in self.nn_config.ma_values])
            x_init = self.x[:, :, 0]
            net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            y_init_and_gradient = net_y(x_init, need_grad=True)
            y_init = y_init_and_gradient[0]
            z = y_init_and_gradient[1]
            
            yl2_batch = tf.reduce_mean(y_init ** 2)
            yl2_ma = tf.get_variable(
                'yl2_ma', [1], TF_DTYPE,
                initializer=tf.constant_initializer(100.0, TF_DTYPE),
                trainable=False)
            yl2 = decay * yl2_ma + (1 - decay) * yl2_batch
            sign = tf.sign(tf.reduce_sum(y_init))
            y = y_init * sign / tf.sqrt(yl2) * self.bsde.L2mean
            y = tf.clip_by_value(y, -5, 5, name=None)
            z = z * sign / tf.sqrt(yl2) * self.bsde.L2mean
            true_z = self.bsde.true_z(x_init)
            normed_true_z = true_z / tf.sqrt(tf.reduce_mean(true_z ** 2))
            error_z = z / tf.sqrt(tf.reduce_mean(z ** 2)) - normed_true_z

            for t in range(0, self.num_time_interval - 1):
                y = y - self.bsde.delta_t * (
                    self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen * y) + \
                    tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                y = tf.clip_by_value(y, -5, 5, name=None)
                yz = net_y(self.x[:, :, t + 1], need_grad=True, reuse=True)
                z = yz[1] * sign / tf.sqrt(yl2) * self.bsde.L2mean
            # terminal time
            y = y - self.bsde.delta_t * (
            self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen * y) + \
                tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)

            y_xT = net_y(self.x[:, :, -1], need_grad=False, reuse=True)
            y_xT = y_xT / tf.sqrt(yl2) * sign * self.bsde.L2mean
            delta = y - y_xT
            # use linear approximation outside the clipped range
            self.train_loss = \
                tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                         2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 100 \
                + tf.nn.relu(2 * self.bsde.L2mean - yl2) * 100

            self.extra_train_ops.append(
                moving_averages.assign_moving_average(yl2_ma, yl2_batch, decay))
            y_hist = net_y(self.x_hist, need_grad=False, reuse=True)
            hist_sign = tf.sign(tf.reduce_sum(y_hist))
            hist_l2 = tf.reduce_mean(y_hist ** 2)
            self.hist_NN = y_hist / tf.sqrt(hist_l2) * hist_sign
            hist_true_y = self.bsde.true_y(self.x_hist)
            self.hist_true = hist_true_y / tf.sqrt(tf.reduce_mean(hist_true_y ** 2))

        true_init = self.bsde.true_y(self.x[:, :, 0])
        rel_err = tf.reduce_mean(tf.square(
            y_init / tf.sqrt(yl2) * sign * self.bsde.L2mean - true_init)) / tf.reduce_mean(
            tf.square(true_init))
        self.init_rel_loss = tf.sqrt(rel_err)

        self.eigen_error = self.eigen - self.bsde.true_eigen
        self.l2 = yl2
        self.grad_error = tf.sqrt(tf.reduce_mean(error_z ** 2) / tf.reduce_mean(true_z ** 2))
        self.NN_consist = tf.constant(0)

        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self.nn_config.lr_boundaries,
                                                    self.nn_config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.train_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')

        all_ops = [apply_op] + self.extra_train_ops
        self.train_ops = tf.group(*all_ops)
        self.t_build = time.time() - start_time

    def build_true(self):
        # both true_y and true_z are multiplied by trainable constant
        start_time = time.time()
        self._init_coef_y = tf.get_variable(
            'init_coef_y', [1], TF_DTYPE, initializer=tf.constant_initializer(0.8, TF_DTYPE),
            trainable=True)
        self._init_coef_z = tf.get_variable(
            'init_coef_z', [1], TF_DTYPE, initializer=tf.constant_initializer(0.8, TF_DTYPE),
            trainable=True)
        with tf.variable_scope('forward'):
            x_init = self.x[:, :, 0]
            y_init = self.bsde.true_y(x_init) * self._init_coef_y
            yl2 = tf.reduce_mean(y_init ** 2)
            y = y_init
            self.z_init = self.bsde.true_z(x_init) * self._init_coef_z
            z = self.z_init
            for t in range(0, self.num_time_interval - 1):
                y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen * y) + \
                    tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                z = self.bsde.true_z(self.x[:, :, t + 1]) * self._init_coef_z
            # terminal time
            y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen * y) + \
                tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)
            y_xT = self.bsde.true_y(self.x[:, :, -1])
            delta = y - y_xT
            # use linear approximation outside the clipped range
            self.train_loss = tf.reduce_mean(
                tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                          2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 100
            hist_true_y = self.bsde.true_y(self.x_hist)
            self.hist_true = hist_true_y / tf.sqrt(tf.reduce_mean(hist_true_y ** 2))
            self.hist_NN = self.hist_true
        self.init_rel_loss = self._init_coef_y -1
        self.eigen_error = self.eigen - self.bsde.true_eigen
        self.l2 = yl2
        self.grad_error = self._init_coef_z -1
        self.NN_consist = tf.constant(0)
        # train operations
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self.nn_config.lr_boundaries,
                                                    self.nn_config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.train_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')

        all_ops = [apply_op] + self.extra_train_ops
        self.train_ops = tf.group(*all_ops)
        self.t_build = time.time() - start_time
        
    


class PeriodNet(object):
    def __init__(self, num_hiddens, out_dim, name='period_net'):
        self.num_hiddens = num_hiddens
        self.out_dim = out_dim
        self.name = name

    def __call__(self, x, need_grad, reuse=tf.AUTO_REUSE):
        dim = x.get_shape()[-1]
        with tf.variable_scope(self.name):
            trig_bases = []
            for i in range(dim+1):
                trig_bases += [tf.sin(i * x), tf.cos(i * x)]
            trig_bases = tf.concat(trig_bases, axis=1)
            h = trig_bases
            for i in range(0, len(self.num_hiddens)):
                h = tf.layers.dense(h, self.num_hiddens[i], activation=tf.nn.relu,
                                    name="full%g" % (i + 1), reuse=reuse)
            u = tf.layers.dense(
                h, self.out_dim, activation=None, name="final_layer", reuse=reuse)
            if need_grad:
                # the shape of grad is 1 x num_sample x dim
                grad = tf.gradients(u, x)[0]
        if need_grad:
            return u, grad*np.sqrt(2)
        else:
            return u
