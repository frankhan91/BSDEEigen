import json
import logging
import os
import numpy as np
import equation as eqn
from utility import get_config, DictionaryUtility
from solver import FeedForwardModel

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('config_path', './configs/sdg_d2_ma.json',
                           """The path to load json file.""")
tf.app.flags.DEFINE_string('exp_name', 'sdg_d2_ma',
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
            if config.eqn_config.eigenpair == 'first':
                model.build()
            else:
                model.build_second()
            result = model.train()
            training_history = result[0]
            # save training history
            np.savetxt('{}_log{}.csv'.format(path_prefix, idx_run),
                       training_history,
                       fmt=['%d', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%d'],
                       delimiter=",",
                       header="step,train_loss, eg_err, egfcn_l2_err, egfcn_infty_err, grad_l2_err, grad_infty_err, consistency_loss, norm_factor, elapsed_time",
                       comments='')
            figure_data = result[1]
            if dim == 1:
                np.savetxt('{}_hist.csv'.format(path_prefix), figure_data, delimiter=",",
                           header="x,y_true,y_NN", comments='')
            else:
                np.savetxt('{}_hist.csv'.format(path_prefix), figure_data, delimiter=",",
                           header="y_true,y_NN", comments='')
            
if __name__ == '__main__':
    main()
