import json
import logging
import os
import numpy as np
#import tensorflow as tf
import equation as eqn
from utility import get_config, DictionaryUtility
from solver import FeedForwardModel
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('config_path', './configs/DoubleWell_d5_second.json',
                           """The path to load json file.""")
tf.app.flags.DEFINE_string('exp_name', 'DoubleWell_d5_second',
                           """The name of numerical experiments.""")
tf.app.flags.DEFINE_integer('num_run', 1,
                            """The number of experiments to repeatedly run for the same problem.""")
tf.app.flags.DEFINE_string('log_dir1', './logs',
                           """Directory where to write event logs and output array.""")


def main():
    config = get_config(FLAGS.config_path)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    if not os.path.exists(FLAGS.log_dir1):
        os.mkdir(FLAGS.log_dir1)
    path_prefix = os.path.join(FLAGS.log_dir1, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(DictionaryUtility.to_dict(config), outfile, indent=2)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-6s %(message)s')

    dim = config.eqn_config.dim

    for idx_run in range(1, FLAGS.num_run+1):
        tf.reset_default_graph()
        with tf.Session() as sess:
            logging.info('Begin to solve %s with run %d' % (FLAGS.exp_name, idx_run))
            model = FeedForwardModel(config, bsde, sess)
            # model.build_linear_consist()
            # model.build_nonlinear_consist()
            # model.build_linear_grad()
            # model.build_nonlinear_grad()
            # model.build_true()
            model.build_double_well()
            result = model.train()
            training_history = result[0]
            # save training history
            np.savetxt('{}_log{}.csv'.format(path_prefix, idx_run),
                       training_history,
                       fmt=['%d', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%d'],
                       delimiter=",",
                       header="step,train_loss, eigen_error, init_rel_loss, init_infty_loss, grad_error, grad_infty_loss, NN_consist,l2, elapsed_time",
                       comments='')
            figure_data = result[1]
            if dim == 1:
                np.savetxt('{}_hist.csv'.format(path_prefix), figure_data, delimiter=",",
                           header="x,y_true,y_NN", comments='')
            else:
                np.savetxt('{}_hist.csv'.format(path_prefix), figure_data, delimiter=",",
                           header="y_true,y_NN", comments='')
            # for 1d
            #print(np.shape(x_hist), np.shape(y_hist), np.shape(np.concatenate((x_hist,y_hist),axis=1)), np.shape(training_history))
            # np.savetxt('{}_hist.csv'.format(path_prefix),np.concatenate((x_hist,y_hist,y_hist_true,y_second),axis=1), delimiter=",",
            #             header="x1_plot,x2_plot,y_plot,y_true,y_true_second", comments='')
            # f = plt.figure()
            # ax1 = f.add_subplot(111)
            # ax1.plot(figure_data[:,1],figure_data[:,3],'ro',label='NN')
            # ax2 = f.add_subplot(111)
            # ax2.plot(x_hist,y_hist_true,'bo', label='true')
            # ax3 = f.add_subplot(111)
            # ax3.plot(x_hist,y_second,'go', label='second')
            
            #plt.hist(y_hist_true, bins='auto')

if __name__ == '__main__':
    main()
