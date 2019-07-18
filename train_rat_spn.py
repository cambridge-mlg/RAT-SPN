import os
import numpy as np
import tensorflow as tf
import datasets
from datasets import DEBD
import argparse
import pickle
import time
import sys
import utils
from models.RegionGraph import RegionGraph
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


def get_num_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        total_parameters += int(np.prod([x.value for x in variable.get_shape()]))
    return total_parameters


def run_training():

    training_start_time = time.time()
    timeout_flag = False

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
    if not ARGS.discrete_leaves:
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
    train_n = int(train_x.shape[0])
    num_dims = int(train_x.shape[1])

    # stores evaluation metrics
    results = {
        'train_ACC': [],
        'train_CE': [],
        'train_LL': [],
        'train_MARG': [],
        'test_ACC': [],
        'test_CE': [],
        'test_LL': [],
        'test_MARG': [],
        'valid_ACC': [],
        'valid_CE': [],
        'valid_LL': [],
        'valid_MARG': [],
        'elapsed_wall_time_epoch': [],
        'best_valid_acc': None,
        'epoch_best_valid_acc': None,
        'best_valid_loss': None,
        'epoch_best_valid_loss': None
    }

    # try to restore model
    latest_model = tf.train.latest_checkpoint(ARGS.result_path + "/checkpoints/")
    if latest_model is not None:
        recovered_epoch = int(latest_model[latest_model.rfind('-') + 1:])

        if not os.path.isfile(ARGS.result_path + '/spn_description.pkl'):
            raise RuntimeError('Found checkpoint, but no description file.')
        if not os.path.isfile(ARGS.result_path + '/results.pkl'):
            raise RuntimeError('Found checkpoint, but no description file.')

        ndo, nco, ARGS_orig, region_graph_layers = pickle.load(open(ARGS.result_path + '/spn_description.pkl', 'rb'))
        if ndo != num_dims or nco != num_classes:
            raise RuntimeError('Inconsistent number of dimensions/classes when trying to retrieve model.')

        results = pickle.load(open(ARGS.result_path + '/results.pkl', "rb"))
        for k in results:
            if type(results[k]) == list and len(results[k]) != recovered_epoch + 1:
                raise AssertionError("Results seem corrupted.")

        # Make Tensorflow model
        rat_spn = RatSpn(region_graph_layers, num_classes, ARGS=ARGS)
        start_epoch_number = recovered_epoch + 1
    else:
        if ARGS.model_description_file:
            ndo, nco, ARGS_orig, region_graph_layers = pickle.load(open(ARGS.model_description_file, 'rb'))
            if ndo != num_dims or nco != num_classes:
                raise RuntimeError('Inconsistent number of dimensions/classes when trying to retrieve model.')

            # Make Tensorflow model
            rat_spn = RatSpn(region_graph_layers, num_classes, ARGS=ARGS)
        else:
            # Make Region Graph
            region_graph = RegionGraph(range(0, num_dims), np.random.randint(0, 1000000000))
            for _ in range(0, ARGS.num_recursive_splits):
                region_graph.random_split(2, ARGS.split_depth)
            region_graph_layers = region_graph.make_layers()

            # Make Tensorflow model
            rat_spn = RatSpn(region_graph_layers, num_classes, ARGS=ARGS)

        if not ARGS.no_save:
            pickle.dump((num_dims, num_classes, ARGS, region_graph_layers), open(ARGS.result_path + '/spn_description.pkl', "wb"))

        start_epoch_number = 0

    # session
    if ARGS.GPU_fraction <= 0.95:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=ARGS.GPU_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.Session()

    # saver
    saver = tf.train.Saver(max_to_keep=ARGS.store_model_max)
    if ARGS.store_best_valid_acc:
        best_valid_acc_saver = tf.train.Saver(max_to_keep=1)
    if ARGS.store_best_valid_loss:
        best_valid_loss_saver = tf.train.Saver(max_to_keep=1)

    # init/load model
    if latest_model is not None:
        saver.restore(sess, latest_model)
        print("")
        print("restored model after epoch {}".format(recovered_epoch))
        print("")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        if ARGS.model_init_file:
            init_saver = tf.train.Saver(rat_spn.all_params)
            init_saver.restore(sess, ARGS.model_init_file)
            print("")
            print("used {} to init model".format(ARGS.model_init_file))
            print("")

    print(rat_spn)
    print("num params: {}".format(get_num_params()))
    print("start training")

    # train_writer = tf.summary.FileWriter("/scratch/rp587/tensorflow_work/", sess.graph)

    ############
    # Training #
    ############

    epoch_elapsed_times = []
    batches_per_epoch = int(np.ceil(float(train_n) / float(ARGS.batch_size)))

    for epoch_n in range(start_epoch_number, ARGS.num_epochs):

        epoch_start_time = time.time()
        rp = np.random.permutation(train_n)

        batch_start_idx = 0
        elapsed_wall_time_epoch = 0.0
        for batch_n in range(0, batches_per_epoch):
            if batch_n + 1 < batches_per_epoch:
                cur_idx = rp[batch_start_idx:batch_start_idx + ARGS.batch_size]
            else:
                cur_idx = rp[batch_start_idx:]
            batch_start_idx += ARGS.batch_size

            feed_dict = {rat_spn.inputs: train_x[cur_idx, :], rat_spn.labels: train_labels[cur_idx]}

            if ARGS.dropout_rate_input is not None:
                feed_dict[rat_spn.dropout_input_placeholder] = ARGS.dropout_rate_input
            if ARGS.dropout_rate_sums is not None:
                feed_dict[rat_spn.dropout_sums_placeholder] = ARGS.dropout_rate_sums

            start_time = time.time()
            if ARGS.optimizer == "em":
                one_hot_labels = -np.inf * np.ones((len(cur_idx), num_classes))
                one_hot_labels[range(len(cur_idx)), [int(x) for x in train_labels[cur_idx]]] = 0.0
                feed_dict[rat_spn.EM_deriv_input_pl] = one_hot_labels

                start_time = time.time()
                sess.run(rat_spn.em_update_accums, feed_dict=feed_dict)
                elapsed_wall_time_epoch += (time.time() - start_time)
            else:
                _, CEM_value, cur_lr, loss_val, ll_mean_val, margin_val = \
                    sess.run([
                        rat_spn.train_op,
                        rat_spn.cross_entropy_mean,
                        rat_spn.learning_rate,
                        rat_spn.objective,
                        rat_spn.neg_norm_ll,
                        rat_spn.neg_margin_objective], feed_dict=feed_dict)
                elapsed_wall_time_epoch += (time.time() - start_time)

                if batch_n % 10 == 1:
                    print("epoch: {}[{}, {:.5f}]   CE: {:.5f}   nll: {:.5f}   negmargin: {:.5f}   loss: {:.5f}   time: {:.5f}".format(
                        epoch_n,
                        batch_n,
                        cur_lr,
                        CEM_value,
                        ll_mean_val,
                        margin_val,
                        loss_val,
                        elapsed_wall_time_epoch))

        if ARGS.optimizer == "em":
            sess.run(rat_spn.em_update_params)
            sess.run(rat_spn.em_reset_accums)
        else:
            sess.run(rat_spn.decrease_lr_op)

        ################
        ### Evaluate ###
        ################
        print('')
        print('epoch {}'.format(epoch_n))

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

        if test_x is not None:
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
        else:
            test_ACC = None
            test_CE = None
            test_LL = None

        if valid_x is not None:
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
        else:
            valid_ACC = None
            valid_CE = None
            valid_LL = None

        print('   ###')
        print('')

        ##############
        ### timing ###
        ##############
        epoch_elapsed_times.append(time.time() - epoch_start_time)
        estimated_next_epoch_time = np.mean(epoch_elapsed_times) + 3 * np.std(epoch_elapsed_times)
        remaining_time = ARGS.timeout_seconds - (time.time() - training_start_time)
        if estimated_next_epoch_time + ARGS.timeout_safety_seconds > remaining_time:
            print("Next epoch might exceed time limit, stop.")
            timeout_flag = True

        if not ARGS.no_save:
            results['train_ACC'].append(train_ACC)
            results['train_CE'].append(train_CE)
            results['train_LL'].append(train_LL)
            results['train_MARG'].append(train_LL)
            results['test_ACC'].append(test_ACC)
            results['test_CE'].append(test_CE)
            results['test_LL'].append(test_LL)
            results['test_MARG'].append(train_LL)
            results['valid_ACC'].append(valid_ACC)
            results['valid_CE'].append(valid_CE)
            results['valid_LL'].append(valid_LL)
            results['valid_MARG'].append(train_LL)
            results['elapsed_wall_time_epoch'].append(elapsed_wall_time_epoch)

            if ARGS.store_best_valid_acc and valid_x is not None:
                if  results['best_valid_acc'] is None or valid_ACC > results['best_valid_acc']:
                    print('Better validation accuracy -> save model')
                    print('')

                    best_valid_acc_saver.save(
                        sess,
                        ARGS.result_path + "/best_valid_acc/model.ckpt",
                        global_step=epoch_n,
                        write_meta_graph=False)

                    results['best_valid_acc'] = valid_ACC
                    results['epoch_best_valid_acc'] = epoch_n

            if ARGS.store_best_valid_loss and valid_x is not None:
                if results['best_valid_loss'] is None or valid_loss < results['best_valid_loss']:
                    print('Better validation loss -> save model')
                    print('')

                    best_valid_loss_saver.save(
                        sess,
                        ARGS.result_path + "/best_valid_loss/model.ckpt",
                        global_step=epoch_n,
                        write_meta_graph=False)

                    results['best_valid_loss'] = valid_loss
                    results['epoch_best_valid_loss'] = epoch_n

            if epoch_n % ARGS.store_model_every_epochs == 0 \
                    or epoch_n + 1 == ARGS.num_epochs \
                    or timeout_flag:
                pickle.dump(results, open(ARGS.result_path + '/results.pkl', "wb"))
                saver.save(sess, ARGS.result_path + "/checkpoints/model.ckpt", global_step=epoch_n, write_meta_graph=False)

        if timeout_flag:
            sys.exit(7)



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
        default=None,
        help='Pickled model description. (%(default)s)'
    )

    infra_arg.add_argument(
        '--model_init_file',
        default=None,
        help='Model file to start from. (%(default)s)'
    )

    infra_arg.add_argument(
        '--result_path',
        default='results/mnist/',
        help='Path where results shall be stored. (%(default)s)'
    )

    infra_arg.add_argument(
        '--store_model_every_epochs',
        default=5,
        help='Number of epochs between subsequent stores of the model. (%(default)s)'
    )

    infra_arg.add_argument(
        '--store_model_max',
        type=int,
        default=1,
        help='Max number of stores. (%(default)s)'
    )

    infra_arg.add_argument(
        '--store_best_valid_acc',
        action='store_true',
        help='Save the model with best validation accuracy. (%(default)s)'
    )

    infra_arg.add_argument(
        '--store_best_valid_loss',
        action='store_true',
        help='Save the model with best validation loss. (%(default)s)'
    )

    infra_arg.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save anything to disc. (%(default)s)'
    )

    infra_arg.add_argument(
        '--GPU_fraction',
        type=float,
        default=1.0,
        help='Fraction of GPU memory to be used by Tensorflow. (%(default)s)'
    )

    infra_arg.add_argument(
        '--timeout_seconds',
        type=float,
        default=1.58e17,
        help="Timeout in seconds. (approx. sun's remaining life time)"
    )

    infra_arg.add_argument(
        '--timeout_safety_seconds',
        type=float,
        default=0.,
        help="Safety margin for timeout in seconds. (%(default)s)"
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

    #
    # optimizer arguments
    #
    optimizer_arg = parser.add_argument_group('Optimizer')

    optimizer_arg.add_argument(
        '--optimizer',
        choices=['adam', 'momentum', 'em'],
        default='adam',
        help='Use Adam, momentum gradient descent, or em. (%(default)s)'
    )

    optimizer_arg.add_argument(
        '--provided_learning_rate',
        type=float,
        default=None,
        help='Learning rate for optimizer. If None, adam: 0.001, momentum: 2.0. (%(default)s)'
    )

    optimizer_arg.add_argument(
        '--num_epochs',
        type=int,
        default=200,
        help='Number of epochs. (%(default)s)'
    )

    optimizer_arg.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size. (%(default)s)'
    )

    optimizer_arg.add_argument(
        '--learning_rate_decay',
        type=float,
        default=0.97,
        help='Decay factor of learning rate. Learning rate is multiplied with this factor after each epoch. '
             'Not applied to adam. (%(default)s)'
    )

    #
    # SPN arguments
    #
    spn_arg = parser.add_argument_group('Sum-Product Network')

    spn_arg.add_argument(
        '--num_recursive_splits',
        type=int,
        default=20,
        help='Number of recursive splits for building the random region graph. (%(default)s)'
    )

    spn_arg.add_argument(
        '--split_depth',
        type=int,
        default=2,
        help='Recursion split depth for building the random region graph. (%(default)s)'
    )

    XORgroup = spn_arg.add_mutually_exclusive_group()
    XORgroup.add_argument(
        '--discrete_leaves',
        dest='discrete_leaves',
        action='store_const',
        const=True,
        default=False,
        help='Shall discrete distributions be used as input distributions? (%(default)s)'
    )
    XORgroup.add_argument(
        '--gaussian_leaves',
        dest='discrete_leaves',
        action='store_const',
        const=False,
        default=False,
        help='Shall Gaussian distributions be used as input distributions?'
    )

    spn_arg.add_argument(
        '--num_states',
        type=int,
        default=2,
        help='Number of states (only relevant for discrete data). (%(default)s)'
    )

    spn_arg.add_argument(
        '--num_input_distributions',
        type=int,
        default=20,
        help='Number of input distributions per regions. (%(default)s)'
    )

    spn_arg.add_argument(
        '--gauss_min_var',
        type=float,
        default=1.0,
        help='Minimum variance of Gaussians (in diagonal covariance). '
             'If gauss_min_var >= gauss_max_var, constant variances == 1 are assumed. (%(default)s)'
    )

    spn_arg.add_argument(
        '--gauss_max_var',
        type=float,
        default=1.0,
        help='Maximum variance of Gaussians (in diagonal covariance). '
             'If gauss_min_var >= gauss_max_var, constant variances == 1 are assumed. (%(default)s)'
    )

    XORgroup = spn_arg.add_mutually_exclusive_group()
    XORgroup.add_argument(
        '--gauss_isotropic',
        dest='gauss_isotropic',
        action='store_const',
        const=True,
        default=True,
        help='Shall Gaussian be isotropic?'
    )
    XORgroup.add_argument(
        '--gauss_nonisotropic',
        dest='gauss_isotropic',
        action='store_const',
        const=False,
        default=True,
        help='Shall Gaussian be nonisotropic? (%(default)s)'
    )

    spn_arg.add_argument(
        '--num_sums',
        type=int,
        default=20,
        help='Number of sum nodes in each region. (%(default)s)'
    )

    XORgroup = spn_arg.add_mutually_exclusive_group()
    XORgroup.add_argument(
        '--normalized_sums',
        dest='normalized_sums',
        action='store_const',
        const=True,
        default=True,
        help='Shall sum-weights be normalized?'
    )
    XORgroup.add_argument(
        '--unnormalized_sums',
        dest='normalized_sums',
        action='store_const',
        const=False,
        default=True,
        help='Shall sum-weights be unnormalized? (%(default)s)'
    )

    spn_arg.add_argument(
        '--dropout_rate_input',
        type=float,
        default=None,
        help='Dropout rate p for inputs. Fraction of (1-p) features are randomly marked as missing. (%(default)s)'
    )

    spn_arg.add_argument(
        '--dropout_rate_sums',
        type=float,
        default=None,
        help='Dropout rates for sum layers. (%(default)s)'
    )

    spn_arg.add_argument(
        '--lambda_discriminative',
        type=float,
        default=1.0,
        help='Tradeoff factor between discriminative and generative objective. '
             'objective: (1-lambda) * log_likelihood + lambda * (kappa * cross_entropy + (1-kappa) * margin)'  
             '(%(default)s)'
    )

    spn_arg.add_argument(
        '--kappa_discriminative',
        type=float,
        default=1.0,
        help='Tradeoff factor between cross-entropy and log-likelihood. '
             'objective: (1-lambda) * log_likelihood + lambda * (kappa * cross_entropy + (1-kappa) * margin)'  
             '(%(default)s)'
    )

    return parser


if __name__ == '__main__':
    parser = make_parser()

    ARGS = parser.parse_args()

    #
    if not ARGS.no_save:
        utils.mkdir_p(ARGS.result_path)

    # set learning rate
    if ARGS.provided_learning_rate is None:
        if ARGS.optimizer == "adam":
            ARGS.provided_learning_rate = 0.001
        elif ARGS.optimizer == "momentum":
            ARGS.provided_learning_rate = 2.0
        elif ARGS.optimizer == "em":
            pass
        else:
            raise NotImplementedError("Unknown optimizer.")

    # process dropout_rate params
    if ARGS.dropout_rate_input is not None:
        if ARGS.dropout_rate_input >= 1.0 or ARGS.dropout_rate_input <= 0.0:
            ARGS.dropout_rate_input = None

        # process dropout_rate params
    if ARGS.dropout_rate_sums is not None:
        if ARGS.dropout_rate_sums >= 1.0 or ARGS.dropout_rate_sums <= 0.0:
            ARGS.dropout_rate_sums = None

    # process lambda_discriminative and kappa_discriminative params
    ARGS.lambda_discriminative = min(max(ARGS.lambda_discriminative, 0.0), 1.0)
    ARGS.kappa_discriminative = min(max(ARGS.kappa_discriminative, 0.0), 1.0)

    # print ARGS
    sorted_keys = sorted(ARGS.__dict__.keys())
    for k in sorted_keys:
        print('{}: {}'.format(k, ARGS.__dict__[k]))
    print("")

    run_training()
