import os
import tempfile
import urllib.request
import shutil
import subprocess
import pickle
import gzip
import numpy as np
import json
import utils


def maybe_download(directory, url_base, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""

    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        print('File already exists: {}'.format(filepath))
        return filepath

    if not os.path.isdir(directory):
        utils.mkdir_p(directory)

    url = url_base + filename
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading {} to {}'.format(url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    print('{} Bytes'.format(os.path.getsize(zipped_filepath)))
    print('Move to {}'.format(filepath))
    shutil.move(zipped_filepath, filepath)
    return filepath


def maybe_download_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        maybe_download('data/mnist', 'http://yann.lecun.com/exdb/mnist/', file)
        print('unzip data/mnist/{}'.format(file))
        filepath = os.path.join('data/mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def maybe_download_fashion_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        maybe_download('data/fashion-mnist', 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', file)
        print('unzip data/fashion-mnist/{}'.format(file))
        filepath = os.path.join('data/fashion-mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def maybe_download_DEBD():
    if os.path.isdir('data/DEBD'):
        print('DEBD already exists')
        return
    subprocess.run(['git', 'clone', 'https://github.com/arranger1044/DEBD', 'data/DEBD'])
    wd = os.getcwd()
    os.chdir('data/DEBD')
    subprocess.run(['git', 'checkout', '80a4906dcf3b3463370f904efa42c21e8295e85c'])
    subprocess.run(['rm', '-rf', '.git'])
    os.chdir(wd)


def maybe_download_imdb(out_path='data/imdb'):
    raw_data_file = os.path.join(out_path, 'imdb.npz')
    word_to_index_file = os.path.join(out_path, 'imdb_word_index.json')

    if os.path.isfile(raw_data_file) and os.path.isfile(word_to_index_file):
        print('Already exists: {}, {}'.format(raw_data_file, word_to_index_file))
        return

    if not os.path.isdir(out_path):
        utils.mkdir_p(out_path)

    if not os.path.isfile(word_to_index_file):
        print('Downloading word_to_index file {}.'.format(word_to_index_file))
        urllib.request.urlretrieve(
            'https://s3.amazonaws.com/text-datasets/imdb_word_index.json',
            word_to_index_file)

    if not os.path.isfile(raw_data_file):
        print('Downloading raw data file {}.'.format(raw_data_file))
        urllib.request.urlretrieve('https://s3.amazonaws.com/text-datasets/imdb.npz', raw_data_file)


def maybe_download_higgs():
    if os.path.isfile('data/higgs/HIGGS.csv'):
        print('Already exists: {}'.format('data/higgs/HIGGS.csv'))
        return

    maybe_download('data/higgs', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/', 'HIGGS.csv.gz')

    print('unzip data/higgs/HIGGS.csv.gz')
    with gzip.open('data/higgs/HIGGS.csv.gz', 'rb') as f_in:
        with open('data/higgs/HIGGS.csv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove('data/higgs/HIGGS.csv.gz')


def maybe_download_all_data():

    print('')
    print('*** Check for mnist ***')
    maybe_download_mnist()

    print('')
    print('*** Check for fashion-mnist ***')
    maybe_download_fashion_mnist()

    print('')
    print('*** Check for DEBD ***')
    maybe_download_DEBD()

    print('')
    print('*** Check for imdb ***')
    maybe_download_imdb()

    print('')
    print('*** Check for theorem ***')
    maybe_download('data/theorem', 'https://www.openml.org/data/get_csv/1587932/phpPbCMyg/', 'theorem.csv')

    print('')
    print('*** Check for higgs ***')
    maybe_download_higgs()

    print('')
    print('*** Check for wine ***')
    maybe_download('data/wine', 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/','winequality-red.csv')
    maybe_download('data/wine', 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/','winequality-white.csv')



########################################################################################################################


def process_imdb(out_path='data/imdb',
             valid_split=0.1,
             max_df=0.8,
             n_words=2000,
             skip_top=20,
             max_words=1000,
             max_topics=200,
             start_char=1,
             oov_char=2,
             rand_gen=1337):
    """Adopted from keras/datasets/imdb/
    """

    out_file = os.path.join(out_path, 'imdb-dense-nmf-{}.pkl'.format(max_topics))
    if os.path.isfile(out_file):
        print('Already exists: {}'.format(out_file))
        return

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    from sklearn.model_selection import train_test_split

    # get word to index dictionary
    word_to_index_file = os.path.join(out_path, 'imdb_word_index.json')
    with open(word_to_index_file) as f:
        word_to_index = json.load(f)

    # get the raw data
    raw_data_file = os.path.join(out_path, 'imdb.npz')
    with np.load(os.path.join(out_path, 'imdb.npz'), allow_pickle=True) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    # pre-processing
    np.random.seed(rand_gen)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + skip_top for w in x] for x in xs]
    elif skip_top:
        xs = [[w + skip_top for w in x] for x in xs]

    if not n_words:
        n_words = max([max(x) for x in xs])

    if oov_char is not None:
        xs = [[w if (skip_top <= w < n_words) else oov_char for w in x]
              for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < n_words]
              for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])
    # end pre-processing

    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    print('Loaded dataset imdb with splits:\n\ttrain\t{}\n\ttest\t{}'.format(x_train.shape, x_test.shape))

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=valid_split,
        random_state=rand_gen)

    assert x_train.shape[0] == y_train.shape[0]
    assert x_valid.shape[0] == y_valid.shape[0]
    print('Splitted for validation into splits:\n\ttrain\t{}\n\tvalid\t{}\n\ttest\t{}'.format(
        x_train.shape,
        x_valid.shape,
        x_test.shape))

    # translating back to words
    print('Replacing word ids back to tokens {}'.format(x_train[:2]))
    index_to_word = [None] * (max(word_to_index.values()) + 1)
    for w, i in word_to_index.items():
        index_to_word[i] = w

    x_train = [' '.join(index_to_word[i] for i in x_train[i] if i < len(index_to_word)) for i in range(x_train.shape[0])]
    x_valid = [' '.join(index_to_word[i] for i in x_valid[i] if i < len(index_to_word)) for i in range(x_valid.shape[0])]
    x_test = [' '.join(index_to_word[i] for i in x_test[i] if i < len(index_to_word)) for i in range(x_test.shape[0])]

    assert len(x_train) == y_train.shape[0]
    assert len(x_valid) == y_valid.shape[0]
    assert len(x_test) == y_test.shape[0]
    print('Done! {}'.format(x_train[:2]))

    # processing into TF-IDF format
    vectorizer = TfidfVectorizer(lowercase=True,
                                 strip_accents='ascii',
                                 stop_words='english',
                                 max_features=max_words,
                                 use_idf=True,
                                 max_df=max_df,
                                 norm=None)

    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_valid = vectorizer.transform(x_valid)
    x_test = vectorizer.transform(x_test)

    assert x_train.shape[0] == y_train.shape[0]
    assert x_valid.shape[0] == y_valid.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    print('TF-IDF shapes:\n\ttrain\t{}\n\tvalid\t{}\n\ttest\t{}'.format(
        x_train.shape,
        x_valid.shape,
        x_test.shape))

    print('After TF-IDF\n {}\n{}\n{}'.format(x_train[:2], x_valid[:2], x_test[:2]))

    data_path = os.path.join(out_path, 'imdb-sparse-tfidf-{}.pklz'.format(x_train.shape[1]))
    with gzip.open(data_path, 'wb') as f:
        pickle.dump((x_train, y_train, x_valid, y_valid, x_test, y_test), f)
    print('Saved to gzipped pickle to {}'.format(data_path))

    nmf = NMF(n_components=max_topics, random_state=rand_gen,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000,
              alpha=.1).fit(x_train)

    x_train = nmf.transform(x_train)
    x_valid = nmf.transform(x_valid)
    x_test = nmf.transform(x_test)

    assert x_train.shape[0] == y_train.shape[0]
    assert x_valid.shape[0] == y_valid.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    print('Final shapes:\n\ttrain\t{}\n\tvalid\t{}\n\ttest\t{}'.format(
        x_train.shape,
        x_valid.shape,
        x_test.shape))

    # saving to pickle
    with open(out_file, 'wb') as f:
        pickle.dump((x_train, y_train, x_valid, y_valid, x_test, y_test), f)
    print('Saved to pickle to {}'.format(out_file))

    # return x_train, y_train, x_valid, y_valid, x_test, y_test


def preprocess_wine(path='data/wine', multi_class=False):

    if multi_class:
        out_file = os.path.join(path, 'wine_multiclass.pkl')
    else:
        out_file = os.path.join(path, 'wine.pkl')

    if os.path.isfile(out_file):
        print('Already exists: {}'.format(out_file))
        return

    np.random.seed(1234567890)

    #
    valid_frac = 0.2
    test_frac = 0.2

    wine_red = np.loadtxt(open(path + "/winequality-red.csv", "rb"), delimiter=";", skiprows=1)
    wine_white = np.loadtxt(open(path + "/winequality-white.csv", "rb"), delimiter=";", skiprows=1)

    wine = np.concatenate((wine_red, wine_white))
    wine_x = wine[:, 0:11]
    wine_labels = wine[:, 11]

    for k in range(11):
        print("#{} = {}".format(k, np.sum(wine_labels == k)))

    print("data shape")
    print(wine_x.shape)

    print("first sample")
    print(wine_x[0, :])
    print(wine_labels[0])

    if multi_class:
        wine_labels_ = wine_labels.astype(int)
        wine_labels_[wine_labels <= 4] = 0
        wine_labels_[wine_labels == 5] = 1
        wine_labels_[wine_labels == 6] = 2
        wine_labels_[wine_labels >= 7] = 3
        wine_labels = wine_labels_
    else:
        wine_labels = (wine_labels >= 6).astype(int)

    unique_labels = np.unique(wine_labels)
    print("")
    print("unique labels")
    print(unique_labels)
    for l in unique_labels:
        print("C{}: {}".format(l, 100 * float(np.sum(wine_labels == l)) / float(len(wine_labels))))

    N = wine_x.shape[0]
    valid_N = int(round(N * valid_frac))
    test_N = int(round(N * test_frac))
    train_N = N - (valid_N + test_N)

    rp = np.random.permutation(N)
    train_x = wine_x[rp[0:train_N], :]
    train_labels = wine_labels[rp[0:train_N]]

    valid_x = wine_x[rp[train_N:(train_N + valid_N)], :]
    valid_labels = wine_labels[rp[train_N:(train_N + valid_N)]]

    test_x = wine_x[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]
    test_labels = wine_labels[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]

    print("")
    print("Train shape")
    print(train_x.shape)
    print(train_labels.shape)
    for l in unique_labels:
        print("C{}: {}".format(l, 100 * float(np.sum(train_labels == l)) / float(len(train_labels))))

    print("")
    print("Valid shape")
    print(valid_x.shape)
    print(valid_labels.shape)
    for l in unique_labels:
        print("C{}: {}".format(l, 100 * float(np.sum(valid_labels == l)) / float(len(valid_labels))))

    print("")
    print("Test shape")
    print(test_x.shape)
    print(test_labels.shape)
    for l in unique_labels:
        print("C{}: {}".format(l, 100 * float(np.sum(test_labels == l)) / float(len(test_labels))))

    with open(out_file, 'wb') as f:
        pickle.dump((train_x, train_labels, valid_x, valid_labels, test_x, test_labels), f)


def preprocess_theorem(path='data/theorem'):

    out_file = os.path.join(path, 'theorem.pkl')

    if os.path.isfile(out_file):
        print('Already exists: {}'.format(out_file))
        return

    np.random.seed(101)

    #
    valid_frac = 0.2
    test_frac = 0.2

    theorem = np.loadtxt(open(os.path.join(path, "theorem.csv"), "rb"), delimiter=",", skiprows=1)
    theorem_x = theorem[:, 0:51]
    theorem_labels = theorem[:, 51]

    for k in range(6):
        theorem_labels[theorem_labels == k + 1] = k
    theorem_labels = theorem_labels.astype(int)

    for k in range(10):
        print("#{} = {}".format(k, np.sum(theorem_labels == k)))

    print("data shape")
    print(theorem_x.shape)

    print("first sample")
    print(theorem_x[0, :])
    print(theorem_labels[0])

    unique_labels = np.unique(theorem_labels)
    print("")
    print("unique labels")
    print(unique_labels)
    for l in unique_labels:
        print("C{}: {}".format(l, 100 * float(np.sum(theorem_labels == l)) / float(len(theorem_labels))))

    N = theorem_x.shape[0]
    valid_N = int(round(N * valid_frac))
    test_N = int(round(N * test_frac))
    train_N = N - (valid_N + test_N)

    rp = np.random.permutation(N)
    train_x = theorem_x[rp[0:train_N], :]
    train_labels = theorem_labels[rp[0:train_N]]

    valid_x = theorem_x[rp[train_N:(train_N + valid_N)], :]
    valid_labels = theorem_labels[rp[train_N:(train_N + valid_N)]]

    test_x = theorem_x[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]
    test_labels = theorem_labels[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]

    print("")
    print("Train shape")
    print(train_x.shape)
    print(train_labels.shape)
    for l in unique_labels:
        print("C{}: {}".format(l, 100 * float(np.sum(train_labels == l)) / float(len(train_labels))))

    print("")
    print("Valid shape")
    print(valid_x.shape)
    print(valid_labels.shape)
    for l in unique_labels:
        print("C{}: {}".format(l, 100 * float(np.sum(valid_labels == l)) / float(len(valid_labels))))

    print("")
    print("Test shape")
    print(test_x.shape)
    print(test_labels.shape)
    for l in unique_labels:
        print("C{}: {}".format(l, 100 * float(np.sum(test_labels == l)) / float(len(test_labels))))

    pickle.dump((train_x, train_labels, valid_x, valid_labels, test_x, test_labels), open(out_file, 'wb'))


def preprocess_higgs(path='data/higgs'):

    out_file = os.path.join(path, 'higgs.pkl')
    if os.path.isfile(out_file):
        print('Already exists: {}'.format(out_file))
        return

    valid_N = 1000000
    test_N = 1000000

    higgs = np.loadtxt(open(path + "/HIGGS.csv", "rb"), delimiter=",", skiprows=0)

    higgs_labels = higgs[:, 0].astype(int)
    higgs_x = higgs[:, 1:]

    N = higgs_x.shape[0]
    train_N = N - (valid_N + test_N)

    rp = np.random.permutation(N)
    train_x = higgs_x[rp[0:train_N], :]
    train_labels = higgs_labels[rp[0:train_N]]

    valid_x = higgs_x[rp[train_N:(train_N + valid_N)], :]
    valid_labels = higgs_labels[rp[train_N:(train_N + valid_N)]]

    test_x = higgs_x[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]
    test_labels = higgs_labels[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]

    print(train_x.shape)
    print(train_labels.shape)

    print(valid_x.shape)
    print(valid_labels.shape)

    print(test_x.shape)
    print(test_labels.shape)

    print(np.unique(train_labels))
    print(np.unique(valid_labels))
    print(np.unique(test_labels))

    mu = np.mean(train_x, 0)
    sigma = np.std(train_x, 0)

    train_x = (train_x - mu) / (sigma + 1e-6)
    valid_x = (valid_x - mu) / (sigma + 1e-6)
    test_x = (test_x - mu) / (sigma + 1e-6)

    pickle.dump((train_x, train_labels, valid_x, valid_labels, test_x, test_labels, mu, sigma), open(out_file, 'wb'))


def preprocess_data():

    print('')
    print('*** Preprocess imdb -- this may take some time ***')
    process_imdb()

    print('')
    print('*** Preprocess wine -- this may take some time ***')
    preprocess_wine()

    print('')
    print('*** Preprocess theorem -- this may take some time ***')
    preprocess_theorem()

    print('')
    print('*** Preprocess higgs -- this may take some time ***')
    preprocess_higgs()


maybe_download_all_data()
preprocess_data()
