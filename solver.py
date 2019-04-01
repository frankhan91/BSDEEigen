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
        # make sure consistent with FBSDE equation
        self.dim = bsde.dim
        self.num_time_interval = bsde.num_time_interval
        self.total_time = bsde.total_time
        # ops for statistics update of batch normalization
        self.extra_train_ops = []
        self.dw = tf.placeholder(TF_DTYPE, [None, self.dim, self.num_time_interval], name='dW')
        self.x = tf.placeholder(TF_DTYPE, [None, self.dim, self.num_time_interval + 1], name='X')
        self.train_loss, self.init_loss, self.init_rel_loss,  self.l2 = None, None, None, None
        self.train_ops, self.t_build = None, None
        self.eigen = tf.get_variable('eigen', shape=[1], dtype=TF_DTYPE,
                                     initializer=tf.random_uniform_initializer(-1, 1), trainable=True)

    def train(self):
        start_time = time.time()
        # to save iteration results
        training_history = []
        # for validation
        dw_valid, x_valid = self.bsde.sample(self.nn_config.valid_size)
        # can still use batch norm of samples in the validation phase
        feed_dict_valid = {self.dw: dw_valid, self.x: x_valid}
        # initialization
        self.sess.run(tf.global_variables_initializer())
        # begin sgd iteration
        for step in range(self.nn_config.num_iterations+1):
            if step % self.nn_config.logging_frequency == 0:
                train_loss, init_loss, init_rel_loss, eigen, l2 = self.sess.run(
                    [self.train_loss, self.init_loss, self.init_rel_loss, self.eigen, self.l2],
                    feed_dict=feed_dict_valid)
                elapsed_time = time.time()-start_time+self.t_build
                training_history.append([step, train_loss, init_loss, init_rel_loss, eigen, l2, elapsed_time])
                if self.nn_config.verbose:
                    logging.info(
                        "step: %5u,    train_loss: %.4e,   init_loss: %.4e, eigen: %.4e, l2: %.4e " % (
                            step, train_loss, init_loss, eigen, l2) +
                        "init_rel_loss: %.4e,   elapsed time %3u" % (
                         init_rel_loss, elapsed_time))
            dw_train, x_train = self.bsde.sample(self.nn_config.batch_size)
            self.sess.run(self.train_ops, feed_dict={self.dw: dw_train, self.x: x_train})
        return np.array(training_history)

    def build(self):
        start_time = time.time()
        with tf.variable_scope('forward'):
            x_init = self.x[:, :, 0]
            net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            net_z = PeriodNet(self.nn_config.num_hiddens, out_dim=self.dim, name='net_z')
            y_init = net_y(x_init)
            z = net_z(x_init)
            yl2 = tf.reduce_mean(y_init ** 2)
            sign = tf.sign(tf.reduce_sum(y_init))
            y_init = y_init / tf.sqrt(yl2) * sign
            y = y_init
            for t in range(0, self.num_time_interval-1):
                y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen *y) + \
                    self.bsde.sigma * tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                z = net_z(self.x[:, :, t + 1], reuse=True)
            # terminal time
            y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen *y) + \
                self.bsde.sigma * tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)

            y_xT = net_y(self.x[:, :, -1], reuse=True) / tf.sqrt(yl2) * sign
            delta = y - y_xT
            # use linear approximation outside the clipped range
            self.train_loss = tf.reduce_mean(
                tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                         2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 100 
                # + tf.reduce_mean(tf.where(tf.abs(grad_y) < DELTA_CLIP, tf.square(grad_y), 2 * DELTA_CLIP * tf.abs(grad_y) - DELTA_CLIP ** 2))
        true_init = self.bsde.true_y(self.x[:, :, 0])
#        g = self.bsde.g_tf(self.x[:, :, 0])
        mask = tf.greater(tf.abs(true_init), 0.1)
        rel_err = tf.abs((y_init - true_init) / true_init)
        rel_err = tf.boolean_mask(rel_err, mask)
        self.init_rel_loss = tf.reduce_mean(rel_err)
        self.init_loss = tf.reduce_mean((true_init - y_init) ** 2)
        self.l2 = tf.reduce_mean((y-1) ** 2)

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

    def build_true(self):
        start_time = time.time()
        with tf.variable_scope('forward'):
            #net_y = PeriodNet(self.nn_config.num_hiddens, out_dim=1, name='net_y')
            x_init = self.x[:, :, 0]
            #y_init = net_y(x_init)
            y_init = self.bsde.true_y(x_init)
            yl2 = tf.reduce_mean(y_init ** 2)
            sign = tf.sign(tf.reduce_sum(y_init))
            y_init = y_init / tf.sqrt(yl2) * sign
            y = y_init
            z = self.bsde.true_z(x_init)
            for t in range(0, self.num_time_interval - 1):
                y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, t], y, z) + self.eigen * y) + \
                    self.bsde.sigma * tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                z = self.bsde.true_z(self.x[:, :, t + 1])
            # terminal time
            y = y - self.bsde.delta_t * (self.bsde.f_tf(self.x[:, :, -2], y, z) + self.eigen * y) + \
                self.bsde.sigma * tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)

            #y_xT = net_y(self.x[:, :, -1], reuse=True) / tf.sqrt(yl2) * sign
            y_xT = self.bsde.true_y(self.x[:, :, -1])
            delta = y - y_xT
            # use linear approximation outside the clipped range
            #self.train_loss = tf.reduce_mean(delta ** 2) * 500
            self.train_loss = tf.reduce_mean(
                tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                          2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 50
                # + tf.reduce_mean(tf.where(tf.abs(grad_y) < DELTA_CLIP, tf.square(grad_y), 2 * DELTA_CLIP * tf.abs(grad_y) - DELTA_CLIP ** 2))
        # f_tf also gives the true eigenfunction
        true_init = self.bsde.true_y(self.x[:, :, 0])
        mask = tf.greater(tf.abs(true_init), 0.1)
        rel_err = tf.abs((y_init - true_init) / true_init)
        rel_err = tf.boolean_mask(rel_err, mask)
        self.init_rel_loss = tf.reduce_mean(rel_err)
        self.init_loss = tf.reduce_mean((true_init - y_init) ** 2)
        self.l2 = tf.reduce_mean((y - 1) ** 2)

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

    def __call__(self, x, reuse=False):
        dim = x.get_shape()[-1]
        with tf.variable_scope(self.name):
            trig_bases = []
            for i in range(dim):
                trig_bases += [tf.sin(i * x), tf.cos(i * x)]
            trig_bases = tf.concat(trig_bases, axis=1)
            h = trig_bases
            for i in range(0, len(self.num_hiddens)):
                h = tf.layers.dense(h, self.num_hiddens[i], activation=tf.nn.relu,
                                    name="full%g" % (i + 1), reuse=reuse)
            u = tf.layers.dense(
                h, self.out_dim, activation=None, name="final_layer", reuse=reuse)
            # grad = tf.gradients(u, x)[0]
        return u
