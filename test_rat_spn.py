import numpy as np
import tensorflow as tf
import datasets
from datasets import DEBD
import argparse
import pickle
from models.RatSpn import RatSpn


def compute_performance(sess, data_x, data_labels, batch_size, spn):
    """Compute some performance measures, X-entropy, likelihood, number of correct samples."""

    num_batches = int(np.ceil(float(data_x.shape[0]) / float(batch_size)))
    test_idx = 0
    num_correct_val = 0
    CE_total = 0.0
    ll = 0.0
    margin = 0.0
    objective = 0.0

    for test_k in range(0, num_batches):
        if test_k + 1 < num_batches:
            batch_data = data_x[test_idx:test_idx + batch_size, :]
            batch_labels = data_labels[test_idx:test_idx + batch_size]
        else:
            batch_data = data_x[test_idx:, :]
            batch_labels = data_labels[test_idx:]

        feed_dict = {spn.inputs: batch_data, spn.labels: batch_labels}
        if spn.dropout_input_placeholder is not None:
            feed_dict[spn.dropout_input_placeholder] = 1.0
        if spn.dropout_sums_placeholder is not None:
            feed_dict[spn.dropout_sums_placeholder ] = 1.0

        num_correct_tmp, CE_tmp, out_tmp, ll_vals, margin_vals, objective_val = sess.run(
            [spn.num_correct,
             spn.cross_entropy,
             spn.outputs,
             spn.log_likelihood,
             spn.log_margin_hinged,
             spn.objective],
            feed_dict=feed_dict)

        num_correct_val += num_correct_tmp
        CE_total += np.sum(CE_tmp)
        ll += np.sum(ll_vals)
        margin += np.sum(margin_vals)
        objective += objective_val * batch_data.shape[0]

        test_idx += batch_size

    ll = ll / float(data_x.shape[0])
    margin = margin / float(data_x.shape[0])
    objective = objective / float(data_x.shape[0])
    
    return num_correct_val, CE_total, ll, margin, objective


def run_testing():

    #############
    # Load data #
    #############
    if ARGS.data_set in ['mnist', 'fashion_mnist']:
        train_x, train_labels, valid_x, valid_labels, test_x, test_labels = datasets.load_mnist(ARGS.data_path)
    elif ARGS.data_set in DEBD:
        train_x, test_x, valid_x = datasets.load_debd(ARGS.data_path, ARGS.data_set)
        train_labels = np.zeros(train_x.shape[0], dtype=np.int32)
        test_labels = np.zeros(test_x.shape[0], dtype=np.int32)
        valid_labels = np.zeros(valid_x.shape[0], dtype=np.int32)
    else:
        if ARGS.data_set == '20ng_classify':
            unpickled = pickle.load(open(ARGS.data_path + '/20ng-50-lda.pkl', "rb"))
        elif ARGS.data_set == 'higgs':
            unpickled = pickle.load(open(ARGS.data_path + '/higgs.pkl', "rb"))
        elif ARGS.data_set == 'wine':
            unpickled = pickle.load(open(ARGS.data_path + '/wine.pkl', "rb"))
        elif ARGS.data_set == 'wine_multiclass':
            unpickled = pickle.load(open(ARGS.data_path + '/wine_multiclass.pkl', "rb"))
        elif ARGS.data_set == 'theorem':
            unpickled = pickle.load(open(ARGS.data_path + '/theorem.pkl', "rb"))
        elif ARGS.data_set == 'imdb':
            unpickled = pickle.load(open(ARGS.data_path + '/imdb-dense-nmf-200.pkl', "rb"))
        train_x = unpickled[0]
        train_labels = unpickled[1]
        valid_x = unpickled[2]
        valid_labels = unpickled[3]
        test_x = unpickled[4]
        test_labels = unpickled[5]

    ######################
    # Data preprocessing #
    ######################
    if ARGS.low_variance_threshold >= 0.0:
        v = np.var(train_x, 0)
        mu = np.mean(v)
        idx = v > ARGS.low_variance_threshold * mu
        train_x = train_x[:, idx]
        test_x = test_x[:, idx]
        if valid_x is not None:
            valid_x = valid_x[:, idx]

    # zero-mean, unit-variance
    if ARGS.normalization == "zmuv":
        train_x_mean = np.mean(train_x, 0)
        train_x_std = np.std(train_x, 0)

        train_x = (train_x - train_x_mean) / (train_x_std + ARGS.zmuv_min_sigma)
        test_x = (test_x - train_x_mean) / (train_x_std + ARGS.zmuv_min_sigma)
        if valid_x is not None:
            valid_x = (valid_x - train_x_mean) / (train_x_std + ARGS.zmuv_min_sigma)

    num_classes = len(np.unique(train_labels))
    num_dims = int(train_x.shape[1])

    ndo, nco, ARGS_orig, region_graph_layers = pickle.load(open(ARGS.model_description_file, 'rb'))
    if ndo != num_dims or nco != num_classes:
        raise RuntimeError('Inconsistent number of dimensions/classes when trying to retrieve model.')

    # Make Tensorflow model
    rat_spn = RatSpn(region_graph_layers, num_classes, ARGS=ARGS_orig)

    # session
    if ARGS.GPU_fraction <= 0.95:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=ARGS.GPU_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.Session()

    # init/load model
    print("Loading model")
    init = tf.global_variables_initializer()
    sess.run(init)
    init_saver = tf.train.Saver(rat_spn.all_params)
    init_saver.restore(sess, ARGS.model_init_file)


    ###########
    # Testing #
    ###########
    print("Run testing")

    num_correct_train, CE_total, train_LL, train_MARG, train_loss = compute_performance(
        sess,
        train_x,
        train_labels,
        100,
        rat_spn)
    train_ACC = 100. * float(num_correct_train) / float(train_x.shape[0])
    train_CE = CE_total / float(train_x.shape[0])
    print('   ###')
    print('   ### accuracy on train set = {}   CE = {}   LL: {}   negmargin: {}'.format(
        train_ACC,
        train_CE,
        train_LL,
        train_MARG))

    num_correct_test, CE_total, test_LL, test_MARG, test_loss = compute_performance(
        sess,
        test_x,
        test_labels,
        100,
        rat_spn)
    test_ACC = 100. * float(num_correct_test) / float(test_x.shape[0])
    test_CE = CE_total / float(test_x.shape[0])
    print('   ###')
    print('   ### accuracy on test set = {}   CE = {}   LL: {}   negmargin: {}'.format(test_ACC, test_CE, test_LL, test_MARG))

    num_correct_valid, CE_total, valid_LL, valid_MARG, valid_loss = compute_performance(
        sess,
        valid_x,
        valid_labels,
        100,
        rat_spn)
    valid_ACC = 100. * float(num_correct_valid) / float(valid_x.shape[0])
    valid_CE = CE_total / float(valid_x.shape[0])
    print('   ###')
    print('   ### accuracy on valid set = {}   CE = {}   LL: {}   margin: {}'.format(
        valid_ACC,
        valid_CE,
        valid_LL,
        valid_MARG))


def make_parser():

    parser = argparse.ArgumentParser()

    #
    # infrastructure arguments
    #
    infra_arg = parser.add_argument_group('Paths')

    infra_arg.add_argument(
        '--data_set',
        choices=['mnist',
                 'fashion_mnist',
                 '20ng_classify',
                 'higgs',
                 'wine',
                 'wine_multiclass',
                 'eeg-eye',
                 'theorem',
                 'imdb'] + DEBD,
        default='mnist',
        help='Path to data. (%(default)s)'
    )

    infra_arg.add_argument(
        '--data_path',
        default='data/mnist/',
        help='Path to data. (%(default)s)'
    )

    infra_arg.add_argument(
        '--model_description_file',
        default='pretrained/mnist/spn_description.pkl',
        help='Pickled model description. (%(default)s)'
    )

    infra_arg.add_argument(
        '--model_init_file',
        default='pretrained/mnist/model.ckpt-171',
        help='Model file to start from. (%(default)s)'
    )

    infra_arg.add_argument(
        '--GPU_fraction',
        type=float,
        default=1.0,
        help='Fraction of GPU memory to be used by Tensorflow. (%(default)s)'
    )

    #
    # preprocessing arguments
    #
    preprop_arg = parser.add_argument_group('Pre-processing')

    preprop_arg.add_argument(
        '--low_variance_threshold',
        type=float,
        default=0.001,
        help='Remove columns whose variance is smaller than low_variance_threshold * mean(variance). (%(default)s)'
    )

    preprop_arg.add_argument(
        '--normalization',
        choices=['none', 'zmuv'],
        default='zmuv',
        help='Which normalization to apply? zmuv: zero-mean-unit-variance (%(default)s)',
    )

    preprop_arg.add_argument(
        '--zmuv_min_sigma',
        type=float,
        default=0.001,
        help='Minimum variance for zmuv normalization (if applied). (%(default)s)',
    )

    return parser


if __name__ == '__main__':
    parser = make_parser()

    ARGS = parser.parse_args()

    run_testing()
