import logging
import time
import numpy as np
import tensorflow as tf

TF_DTYPE = tf.float64
MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


class FeedForwardModel(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde, sess):
        self._eqn_config = config.eqn_config
        self._nn_config = config.nn_config
        self._bsde = bsde
        self._sess = sess
        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time
        # ops for statistics update of batch normalization
        self._extra_train_ops = []
        self._init_coeff = tf.constant([1.0])

    def train(self):
        start_time = time.time()
        # to save iteration results
        training_history = []
        # for validation
        dw_valid, x_valid = self._bsde.sample(self._nn_config.valid_size)
        # can still use batch norm of samples in the validation phase
        feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}
        # initialization
        self._sess.run(tf.global_variables_initializer())
        # begin sgd iteration
        for step in range(self._nn_config.num_iterations+1):
            if step % self._nn_config.logging_frequency == 0:
                train_loss, init_loss, init_rel_loss, init_coeff = self._sess.run(
                    [self._train_loss, self._init_loss, self._init_rel_loss, self._init_coeff],
                    feed_dict=feed_dict_valid)
                elapsed_time = time.time()-start_time+self._t_build
                training_history.append([step, train_loss, init_loss, init_rel_loss, elapsed_time])
                if self._nn_config.verbose:
                    logging.info(
                        "step: %5u,    train_loss: %.4e,   init_loss: %.4e, init_rel_loss: %.4e,  init_coeff: %.2f   elapsed time %3u" % (
                        step, train_loss, init_loss, init_rel_loss, init_coeff, elapsed_time))
            dw_train, x_train = self._bsde.sample(self._nn_config.batch_size)
            self._sess.run(self._train_ops, feed_dict={self._dw: dw_train,
                                                       self._x: x_train,
                                                       self._is_training: True})
        return np.array(training_history)

    def build(self):
        start_time = time.time()
        self._dw = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval], name='dW')
        self._x = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval + 1], name='X')
        self._is_training = tf.placeholder(tf.bool)
        num_hiddens = self._nn_config.num_hiddens.copy()
        num_hiddens[-1] = 1  # the dimension of y_init is 1 rather than dim
        x_init = self._x[:, :, 0]
        self._y_init = self._subnetwork(x_init, num_hiddens, 'y_init')
        grad_init = tf.gradients(self._y_init, x_init)[0]
        y = self._y_init
        z = grad_init
        with tf.variable_scope('forward'):
            for t in range(0, self._num_time_interval-1):
                if t == 0:
                    reuse = False
                else:
                    reuse = True
                y = y - self._bsde.delta_t * self._bsde.f_tf(self._x[:, :, t], y, z) + \
                    tf.reduce_sum(z * self._dw[:, :, t], 1, keepdims=True)
                z = self._subnetwork(
                    self._x[:, :, t + 1], self._nn_config.num_hiddens,
                    name="z_net", reuse=reuse)
            # terminal time
            y = y - self._bsde.delta_t * self._bsde.f_tf(self._x[:, :, -2], y, z) + \
                tf.reduce_sum(z * self._dw[:, :, -1], 1, keepdims=True)
            delta = y - self._bsde.g_tf(self._x[:, :, -1])
            # use linear approximation outside the clipped range
            self._train_loss = tf.reduce_mean(
                tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                         2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)) * 10
        # f_tf also gives the true eigenfunction
        true_init = self._bsde.g_tf(self._x[:, :, 0])
        mask = tf.greater(tf.abs(true_init), 0.1)
        rel_err = tf.abs((self._y_init - true_init) / true_init)
        rel_err = tf.boolean_mask(rel_err, mask)
        self._init_rel_loss = tf.reduce_mean(rel_err)
        self._init_loss = tf.reduce_mean((true_init - self._y_init) ** 2)

        # train operations
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self._nn_config.lr_boundaries,
                                                    self._nn_config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')
        all_ops = [apply_op] + self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time()-start_time

    def build_true(self):
        start_time = time.time()
        self._dw = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval], name='dW')
        self._x = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval + 1], name='X')
        self._is_training = tf.placeholder(tf.bool)
        self._init_coeff = tf.get_variable(
            'init_coeff', [1], TF_DTYPE, initializer=tf.constant_initializer(0.8, TF_DTYPE),
            trainable=True)
        self._y_init = self._bsde.g_tf(self._x[:, :, 0]) * self._init_coeff
        y = self._y_init
        z = self._bsde.true_z(self._x[:, :, 0])
        with tf.variable_scope('forward'):
            for t in range(0, self._num_time_interval-1):
                y = y - self._bsde.delta_t * self._bsde.f_tf(self._x[:, :, t], y, z) + \
                    tf.reduce_sum(z * self._dw[:, :, t], 1, keepdims=True)
                z = self._bsde.true_z(self._x[:, :, t+1])
            # terminal time
            y = y - self._bsde.delta_t * self._bsde.f_tf(self._x[:, :, -2], y, z) + \
                tf.reduce_sum(z * self._dw[:, :, -1], 1, keepdims=True)
            delta = y - self._bsde.g_tf(self._x[:, :, -1])
        # use linear approximation outside the clipped range
        self._train_loss = tf.reduce_mean(
            tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                     2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        # f_tf also gives the true eigenfunction
        true_init = self._bsde.g_tf(self._x[:, :, 0])
        mask = tf.greater(tf.abs(true_init), 0.3)
        rel_err = tf.abs((self._y_init - true_init) / true_init)
        rel_err = tf.boolean_mask(rel_err, mask)
        self._init_rel_loss = tf.reduce_mean(rel_err)
        self._init_loss = tf.reduce_mean((true_init - self._y_init) ** 2)

        # train operations
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self._nn_config.lr_boundaries,
                                                    self._nn_config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')
        all_ops = [apply_op] + self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time()-start_time

    def _subnetwork(self, x, num_hiddens, name, reuse=False):
        with tf.variable_scope(name):
            # map x to something periodic first
            trig_bases = []
            for i in range(self._dim):
                trig_bases += [tf.sin(i*x), tf.cos(i*x)]
            trig_bases = tf.concat(trig_bases, axis=1)
            hiddens = trig_bases
            for i in range(1, len(num_hiddens)-1):
                hiddens = tf.layers.dense(
                    hiddens, num_hiddens[i], activation=tf.nn.relu, name="full%g" % (i + 1), reuse=reuse)
            output = tf.layers.dense(
                hiddens, num_hiddens[-1], activation=None, name="final_layer", reuse=reuse)
        return output