import json
import logging
import os
import numpy as np
import tensorflow as tf
import equation as eqn
from utility import get_config, DictionaryUtility
from solver import FeedForwardModel
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('config_path', './configs/Sdg_d5_ma.json',
                           """The path to load json file.""")
tf.app.flags.DEFINE_string('exp_name', 'sdg_d5',
                           """The name of numerical experiments.""")
tf.app.flags.DEFINE_integer('num_run', 1,
                            """The number of experiments to repeatedly run for the same problem.""")
tf.app.flags.DEFINE_string('log_dir', './logs',
                           """Directory where to write event logs and output array.""")


def main():
    config = get_config(FLAGS.config_path)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(DictionaryUtility.to_dict(config), outfile, indent=2)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-6s %(message)s')

    for idx_run in range(1, FLAGS.num_run+1):
        tf.reset_default_graph()
        with tf.Session() as sess:
            logging.info('Begin to solve %s with run %d' % (FLAGS.exp_name, idx_run))
            model = FeedForwardModel(config, bsde, sess)
            model.build_linear_consist()
            # model.build_nonlinear_consist()
            #model.build_linear_grad()
            #model.build_nonlinear_grad()
            #model.build_true()
            result = model.train()
            training_history = result[0]
            # save training history
            np.savetxt('{}_log{}.csv'.format(path_prefix, idx_run),
                       training_history,
                       fmt=['%d', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%d'],
                       delimiter=",",
                       header="step,train_loss, eigen_error, init_rel_loss, grad_error, NN_consist,l2, elapsed_time",
                       comments='')
            # y_hist_NN = result[1]
            # y_hist_true = result[2]
            # np.savetxt('{}HistTruey.txt'.format(path_prefix),y_hist_true)
            # np.savetxt('{}HistNN.txt'.format(path_prefix),y_hist_NN)
            # print(np.histogram(y_hist_true))
            # print(np.histogram(y_hist_NN))
            # plt.hist(y_hist_true, bins='auto')
            # plt.hist(result[1], bins='auto')


if __name__ == '__main__':
    main()
