import logging
import time
import numpy as np
import tensorflow as tf

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
        self.x_input_fake = np.zeros(
            shape=[self.nn_config.batch_size, self.dim, self.num_time_interval+1])
        self.train_loss, self.eigen_error, self.init_rel_loss, self.NN_consist, self.eqn_error,self.l2 = None, None, None, None, None, None
        self.train_ops, self.t_build = None, None
        self.eigen = tf.get_variable('eigen', shape=[1], dtype=TF_DTYPE,
                                     initializer=tf.random_uniform_initializer(-3.001, -3.0), trainable=True)
        #self.net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
        #self.net_y_copy = self.net_y

    def train(self):
        start_time = time.time()
        # to save iteration results
        training_history = []
        # for validation
        #dw_valid, x_valid = self.bsde.sample_general_old(self.nn_config.valid_size)
        dw_valid, x_valid = self.bsde.sample_uniform(self.nn_config.valid_size)
        # can still use batch norm of samples in the validation phase
        feed_dict_valid = {self.dw: dw_valid, self.x: x_valid}
        # initialization
        self.sess.run(tf.global_variables_initializer())
        # begin sgd iteration
        for step in range(self.nn_config.num_iterations+1):
            if step % self.nn_config.logging_frequency == 0:
                train_loss, eigen_error, init_rel_loss, grad_error, NN_consist,eqn_error,l2 = self.sess.run(
                    [self.train_loss, self.eigen_error, self.init_rel_loss, self.grad_error, self.NN_consist, self.eqn_error,self.l2],
                    feed_dict=feed_dict_valid)
                elapsed_time = time.time()-start_time+self.t_build
                training_history.append([step, train_loss, eigen_error, init_rel_loss, grad_error,NN_consist,eqn_error, l2, elapsed_time])
                if self.nn_config.verbose:
                    logging.info(
                        "step: %5u,    train_loss: %.4e,   eigen_error: %.4e, grad_error: %.4e, NN_consist: %.4e, eqn_error: %.4e, l2: %.4e " % (
                            step, train_loss, eigen_error, grad_error, NN_consist, eqn_error, l2) +
                        "init_rel_loss: %.4e,   elapsed time %3u" % (
                         init_rel_loss, elapsed_time))
            # use function for sampling, could be bsde.true_y_np or self.y_init_func
#            if step % 10 == 0 & step >= 9000:
#                self.net_y_copy = self.net_y
            #dw_train, x_train = self.bsde.sample_general_new(self.nn_config.batch_size,
            #                                              sample_func=self.y_init_func)
            dw_train, x_train = self.bsde.sample_uniform(self.nn_config.batch_size)
            #dw_train, x_train = self.bsde.sample_general_old(self.nn_config.batch_size)
#            if step < 2000:
#                self.sess.run(self.train_ops0, feed_dict={self.dw: dw_train, self.x: x_train})
#            else:
#                self.sess.run(self.train_ops, feed_dict={self.dw: dw_train, self.x: x_train})
            self.sess.run(self.train_ops, feed_dict={self.dw: dw_train, self.x: x_train})
        return np.array(training_history)

    def build(self):
        start_time = time.time()
        with tf.variable_scope('forward'):
            x_init = self.x[:, :, 0]
            net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            #self.net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            net_z = PeriodNet(self.nn_config.num_hiddens, out_dim=self.dim, name='net_z')
            y_init_and_gradient = net_y(x_init,need_grad=True)
            y_init = y_init_and_gradient[0]
            grad_y = y_init_and_gradient[1]
            z = net_z(x_init, need_grad=False)
            #z_init = z
            yl2 = tf.reduce_mean(y_init ** 2)
            true_z = self.bsde.true_z(x_init)
            #y_init_unnormalized = y_init
            sign = tf.sign(tf.reduce_sum(y_init))
            normed_true_z = true_z / tf.sqrt(tf.reduce_mean(true_z ** 2))
            error_z = z / tf.sqrt(tf.reduce_mean(z ** 2)) - normed_true_z
            y_init = y_init / tf.sqrt(yl2) * sign
            #NN_consist = z - grad_y / tf.sqrt(yl2) * sign
            NN_consist = grad_y * sign /tf.sqrt(tf.reduce_mean(grad_y ** 2)) - normed_true_z
            self.y_init = y_init
            y = y_init
#            temp1 = tf.reduce_mean(tf.abs(y))
#            y_init_copy  = self.net_y_copy(x_init,need_grad=False)
#            yl2_copy = tf.reduce_mean(y_init_copy ** 2)
#            self.y_init_copy = y_init_copy / tf.sqrt(yl2_copy)
            
            for t in range(0, self.num_time_interval-1):
                y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen *y) + \
                    tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                z = net_z(self.x[:, :, t + 1], need_grad=False, reuse=True)
#                if t==3:
#                    temp2 = tf.reduce_mean(tf.abs(y))
#                if t==6:
#                    temp3 = tf.reduce_mean(tf.abs(y))
#                if t==9:
#                    temp4 = tf.reduce_mean(tf.abs(y))
            # terminal time
            y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen *y) + \
                tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)
            
            y_xT = net_y(self.x[:, :, -1], need_grad=False, reuse=True)
            y_xT = y_xT / tf.sqrt(yl2) * sign
            
            delta = y - y_xT
            # use linear approximation outside the clipped range
            
            self.train_loss = tf.reduce_mean(
                tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                         2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 100
        true_init = self.bsde.true_y(self.x[:, :, 0])
        #self.train_loss0 = tf.reduce_mean(tf.square(y_init_unnormalized - true_init))\
        #    + tf.reduce_mean(tf.square(z_init - true_z))
        true_init = true_init / tf.sqrt(tf.reduce_mean(true_init ** 2))
        mask = tf.greater(tf.abs(true_init), 0.1)
        rel_err = tf.abs((y_init - true_init) / true_init)
        rel_err = tf.boolean_mask(rel_err, mask)
        #error_y = y_init - true_init
        #self.init_rel_loss = tf.sqrt(tf.reduce_mean(error_y ** 2))
        self.init_rel_loss = tf.reduce_mean(rel_err)
        self.eigen_error = self.eigen - self.bsde.true_eigen
        self.l2 = yl2
        self.grad_error = tf.sqrt(tf.reduce_mean(error_z ** 2))
        self.NN_consist = tf.sqrt(tf.reduce_mean(NN_consist ** 2))
        self.eqn_error = tf.constant(0)
#        self.eigen_error = temp1
#        self.grad_error = temp2
#        self.NN_consist = temp3
#        self.init_rel_loss = temp4
        
        # train operations
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self.nn_config.lr_boundaries,
                                                    self.nn_config.lr_values)
        trainable_variables = tf.trainable_variables()
        #yl2 = tf.stop_gradient(yl2)
        grads = tf.gradients(self.train_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')
        
        all_ops = [apply_op] + self.extra_train_ops
        self.train_ops = tf.group(*all_ops)
        
#        grads0 = tf.gradients(self.train_loss0, trainable_variables)
#        optimizer0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
#        apply_op0 = optimizer0.apply_gradients(zip(grads0, trainable_variables),
#                                             global_step=global_step, name='train0_step')
#        
#        all_ops0 = [apply_op0] + self.extra_train_ops
#        self.train_ops0 = tf.group(*all_ops0)
        
        self.t_build = time.time() - start_time

    def build_true(self):
        start_time = time.time()
        with tf.variable_scope('forward'):
            #net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            x_init = self.x[:, :, 0]
            #y_init = net_y(x_init)
            y_init = self.bsde.true_y(x_init)
            self.y_init = y_init
            yl2 = tf.reduce_mean(y_init ** 2)
            y = y_init
            z = self.bsde.true_z(x_init)
            for t in range(0, self.num_time_interval - 1):
                y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen * y) + \
                    tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                z = self.bsde.true_z(self.x[:, :, t + 1])
            # terminal time
            y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen * y) + \
                tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)

            #y_xT = net_y(self.x[:, :, -1], reuse=True) / tf.sqrt(yl2) * sign
            y_xT = self.bsde.true_y(self.x[:, :, -1])
            delta = y - y_xT
            # use linear approximation outside the clipped range
            #self.train_loss = tf.reduce_mean(delta ** 2) * 500
            self.train_loss = tf.reduce_mean(
                tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                          2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 100
                # + tf.reduce_mean(tf.where(tf.abs(grad_y) < DELTA_CLIP, tf.square(grad_y), 2 * DELTA_CLIP * tf.abs(grad_y) - DELTA_CLIP ** 2))
        # f_tf also gives the true eigenfunction
        true_init = self.bsde.true_y(self.x[:, :, 0])
        mask = tf.greater(tf.abs(true_init), 0.1)
        rel_err = tf.abs((y_init - true_init) / true_init)
        rel_err = tf.boolean_mask(rel_err, mask)
        self.init_rel_loss = tf.reduce_mean(rel_err)
        self.eigen_error = self.eigen - self.bsde.true_eigen
        self.l2 = yl2
        self.grad_error = tf.constant(0)
        self.NN_consist = tf.constant(0)
        self.eqn_error = tf.constant(0)
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

    def build_unnorm(self):
        start_time = time.time()
        with tf.variable_scope('forward'):
            x_init = self.x[:, :, 0]
            net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            #self.net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            net_z = PeriodNet(self.nn_config.num_hiddens, out_dim=self.dim, name='net_z')
            y_init_and_gradient = net_y(x_init,need_grad=True)
            y_init = y_init_and_gradient[0]
            grad_y = y_init_and_gradient[1]
            z = net_z(x_init, need_grad=False)
            x_init_split = tf.split(x_init, self.dim, axis=1)
            x_init_concat = tf.concat(x_init_split, axis=1)
            #z_and_gradient = net_z(x_init, need_grad=True)
            #z = z_and_gradient[0]
            #Hess = z_and_gradient[1]
            z_init = net_z(x_init_concat, need_grad=False)
            Laplician = []
            for i in range(self.dim):
                Laplician.append(tf.reshape( tf.gradients(z_init[:,i], x_init_split[i]), [-1,1]))
            Laplician = tf.reduce_sum(tf.concat(Laplician, axis=1), axis = 1) / self.bsde.sigma
#            Laplician = []
#            y0 = net_y(x_init_concat,need_grad=True)
#            y1 = y0[1]
#            for i in range(self.dim):
#                Laplician.append(tf.reshape( tf.gradients(y1[:,i], x_init_split[i]), [-1,1]))
#            Laplician = tf.reduce_sum(tf.concat(Laplician, axis=1), axis = 1) / self.bsde.sigma
            
            
            yl2 = tf.reduce_mean(y_init ** 2)
            true_z = self.bsde.true_z(x_init)
            #y_init_unnormalized = y_init
            sign = tf.sign(tf.reduce_sum(y_init))
            #normed_true_z = true_z / tf.sqrt(tf.reduce_mean(true_z ** 2))
            error_z = z - true_z
            #error_z = z / tf.sqrt(tf.reduce_mean(z ** 2)) - normed_true_z
            #y_init = y_init / tf.sqrt(yl2) * sign
            NN_consist = z - grad_y * sign
            #NN_consist = grad_y * sign /tf.sqrt(tf.reduce_mean(grad_y ** 2)) - normed_true_z
            self.y_init = y_init
            y = y_init
            # recall that f_tf = -V*y - epsl*y^3, the eqn is -\Delta psi - f = lambda psi
            self.eqn_error = tf.reduce_mean(tf.abs(- Laplician - self.bsde.f_tf(x_init, y_init, z) - self.eigen * y_init))
#            y_init_copy  = self.net_y_copy(x_init,need_grad=False)
#            yl2_copy = tf.reduce_mean(y_init_copy ** 2)
#            self.y_init_copy = y_init_copy / tf.sqrt(yl2_copy)
            y = y * self.bsde.norm_const / yl2
            for t in range(0, self.num_time_interval-1):
                y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen *y) + \
                    tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                z = net_z(self.x[:, :, t + 1], need_grad=False, reuse=True)
            # terminal time
            y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen *y) + \
                tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)
            
            y_xT = net_y(self.x[:, :, -1], need_grad=False, reuse=True) * sign
            y_xT = y_xT / tf.sqrt(yl2) * sign * self.bsde.norm_const
#            yTl2 = tf.reduce_mean(y ** 2)
#            yXTl2 = tf.reduce_mean(y_xT ** 2)
#            delta = y/yTl2 - y_xT/yXTl2
            delta = y - y_xT
            
            # use linear approximation outside the clipped range
            
        
            self.train_loss = tf.reduce_mean(
                tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                         2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 100 \
                    + tf.nn.relu(0.2 - tf.reduce_mean(tf.abs(y_init))) * 100
        true_init = self.bsde.true_y(self.x[:, :, 0])
        self.train_loss0 = tf.reduce_mean(tf.square(y_init - true_init))\
            + tf.reduce_mean(tf.square(error_z))
        #true_init = true_init / tf.sqrt(tf.reduce_mean(true_init ** 2))
        mask = tf.greater(tf.abs(true_init), 0.1)
        rel_err = tf.abs((y_init - true_init) / true_init)
        rel_err = tf.boolean_mask(rel_err, mask)
        #error_y = y_init - true_init
        #self.init_rel_loss = tf.sqrt(tf.reduce_mean(error_y ** 2))
        self.init_rel_loss = tf.reduce_mean(rel_err)
        self.eigen_error = self.eigen - self.bsde.true_eigen
        self.l2 = yl2
        self.grad_error = tf.sqrt(tf.reduce_mean(error_z ** 2))
        self.NN_consist = tf.sqrt(tf.reduce_mean(NN_consist ** 2))
        
#        self.eigen_error = yTl2
#        self.grad_error = yXTl2
#        self.NN_consist = temp3
#        self.init_rel_loss = temp4        
        
        
        # train operations
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self.nn_config.lr_boundaries,
                                                    self.nn_config.lr_values)
        trainable_variables = tf.trainable_variables()
        #yl2 = tf.stop_gradient(yl2)
        grads = tf.gradients(self.train_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')
        
        all_ops = [apply_op] + self.extra_train_ops
        self.train_ops = tf.group(*all_ops)
        
        grads0 = tf.gradients(self.train_loss0, trainable_variables)
        optimizer0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op0 = optimizer0.apply_gradients(zip(grads0, trainable_variables),
                                             global_step=global_step, name='train0_step')
        
        all_ops0 = [apply_op0] + self.extra_train_ops
        self.train_ops0 = tf.group(*all_ops0)
        
        self.t_build = time.time() - start_time
    
    def build_true_y(self):
        start_time = time.time()
        with tf.variable_scope('forward'):
            #net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            x_init = self.x[:, :, 0]
            true_z = self.bsde.true_z(x_init)
            #y_init = net_y(x_init)
            y_init = self.bsde.true_y(x_init)
            self.y_init = y_init
            yl2 = tf.reduce_mean(y_init ** 2)
            y = y_init
            net_z = PeriodNet(self.nn_config.num_hiddens, out_dim=self.dim, name='net_z')
            z = net_z(x_init, need_grad=False)
            normed_true_z = true_z / tf.sqrt(tf.reduce_mean(true_z ** 2))
            error_z = z / tf.sqrt(tf.reduce_mean(z ** 2)) - normed_true_z
            for t in range(0, self.num_time_interval - 1):
                y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen * y) + \
                    tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                z = net_z(self.x[:, :, t + 1], need_grad=False, reuse=True)
            # terminal time
            y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen * y) + \
                tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)

            #y_xT = net_y(self.x[:, :, -1], reuse=True) / tf.sqrt(yl2) * sign
            y_xT = self.bsde.true_y(self.x[:, :, -1])
            delta = y - y_xT
            # use linear approximation outside the clipped range
            #self.train_loss = tf.reduce_mean(delta ** 2) * 500
            self.train_loss = tf.reduce_mean(
                tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                          2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 100
                # + tf.reduce_mean(tf.where(tf.abs(grad_y) < DELTA_CLIP, tf.square(grad_y), 2 * DELTA_CLIP * tf.abs(grad_y) - DELTA_CLIP ** 2))
        # f_tf also gives the true eigenfunction
        true_init = self.bsde.true_y(self.x[:, :, 0])
        mask = tf.greater(tf.abs(true_init), 0.1)
        rel_err = tf.abs((y_init - true_init) / true_init)
        rel_err = tf.boolean_mask(rel_err, mask)
        self.init_rel_loss = tf.reduce_mean(rel_err)
        self.eigen_error = self.eigen - self.bsde.true_eigen
        self.l2 = yl2
        self.grad_error = tf.sqrt(tf.reduce_mean(error_z ** 2))
        self.NN_consist = tf.constant(0)
        self.eqn_error = tf.constant(0)
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
        
    def y_init_func(self, x):
        self.x_input_fake[:, :, 0] = x
        return self.sess.run(self.y_init, feed_dict={self.x: self.x_input_fake})


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
#                res = tf.layers.dense(h, self.num_hiddens[i], activation=tf.nn.relu,
#                                    name="full%g" % (i + 1), reuse=reuse)
#                if i == 0:
#                    h = res
#                else:
#                    h += res
            u = tf.layers.dense(
                h, self.out_dim, activation=None, name="final_layer", reuse=reuse)
            if need_grad:
                grad = tf.gradients(u, x)[0]
        if need_grad:
            return u, grad*np.sqrt(2)
        else:
            return u
