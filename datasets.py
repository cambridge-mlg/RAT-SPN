import numpy as np
import os
import csv


DEBD = ['accidents',
        'ad',
        'baudio',
        'bbc',
        'bnetflix',
        'book',
        'c20ng',
        'cr52',
        'cwebkb',
        'dna',
        'jester',
        'kdd',
        'kosarek',
        'msnbc',
        'msweb',
        'nltcs',
        'plants',
        'pumsb_star',
        'tmovie',
        'tretail']


DEBD_num_vars = {
    'accidents': 111,
    'ad': 1556,
    'baudio': 100,
    'bbc': 1058,
    'bnetflix': 100,
    'book': 500,
    'c20ng': 910,
    'cr52': 889,
    'cwebkb': 839,
    'dna': 180,
    'jester': 100,
    'kdd': 64,
    'kosarek': 190,
    'msnbc': 17,
    'msweb': 294,
    'nltcs': 16,
    'plants': 69,
    'pumsb_star': 163,
    'tmovie': 500,
    'tretail': 135}


DEBD_display_name = {
    'accidents': 'accidents',
    'ad': 'ad',
    'baudio': 'audio',
    'bbc': 'bbc',
    'bnetflix': 'netflix',
    'book': 'book',
    'c20ng': '20ng',
    'cr52': 'reuters-52',
    'cwebkb': 'web-kb',
    'dna': 'dna',
    'jester': 'jester',
    'kdd': 'kdd-2k',
    'kosarek': 'kosarek',
    'msnbc': 'msnbc',
    'msweb': 'msweb',
    'nltcs': 'nltcs',
    'plants': 'plants',
    'pumsb_star': 'pumsb-star',
    'tmovie': 'each-movie',
    'tretail': 'retail'}


def load_mnist(data_dir):
    """Load MNIST"""

    # save current random state
    state = np.random.get_state()
    np.random.seed(12345)

    # make train/validation split
    validation_frac = 0.1
    num_valid = max(int(round(60000 * validation_frac)), 1)
    rp = np.random.permutation(60000)
    valid_idx = sorted(rp[0:num_valid])
    train_idx = sorted(rp[num_valid:])

    # restore random state
    np.random.set_state(state)

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    valid_x = train_x[valid_idx, :]
    valid_labels = train_labels[valid_idx]
    train_x = train_x[train_idx, :]
    train_labels = train_labels[train_idx]

    return train_x, train_labels, valid_x, valid_labels, test_x, test_labels


def load_debd(data_dir, name, dtype='int32'):
    """Load one of the twenty binary density esimtation benchmark datasets."""

    train_path = os.path.join(data_dir, 'datasets', name, name + '.train.data')
    test_path = os.path.join(data_dir, 'datasets', name, name + '.test.data')
    valid_path = os.path.join(data_dir, 'datasets', name, name + '.valid.data')

    reader = csv.reader(open(train_path, 'r'), delimiter=',')
    train_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(test_path, 'r'), delimiter=',')
    test_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(valid_path, 'r'), delimiter=',')
    valid_x = np.array(list(reader)).astype(dtype)

    return train_x, test_x, valid_x

