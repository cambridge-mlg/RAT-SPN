import numpy as np
import pickle
import os

datasets = ['mnist', 'fashion-mnist', 'wine', 'theorem', 'higgs', 'imdb']
result_basefolder = 'results/ratspn/'


def evaluate():

    ls = os.listdir(result_basefolder)

    for dataset in datasets:
        print()
        if dataset not in ls:
            print('Results for {} not found.'.format(dataset))
            continue

        ls2 = os.listdir(os.path.join(result_basefolder, dataset))

        best_valid_acc = -np.inf
        best_test_acc = -np.inf
        best_model = None
        best_epoch = None
        test_accs = []

        for result_folder in ls2:

            argdict = {}
            for a in result_folder.split('__'):
                last_ = a.rfind('_')
                argdict[a[:last_]] = float(a[last_ + 1:])

            try:
                results = pickle.load(open('{}/{}/{}/results.pkl'.format(
                    result_basefolder,
                    dataset,
                    result_folder), "rb"))
            except:
                print()
                print("can't load")
                print(result_folder)
                continue

            valid_acc = results['best_valid_acc']
            test_acc = results['test_ACC'][results['epoch_best_valid_acc']]
            test_accs.append(test_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_test_acc = test_acc
                best_model = argdict
                best_epoch = results['epoch_best_valid_acc']

        print('Test accuracy: {}'.format(best_test_acc))
        print('Achieved by configuration:')
        print(best_model)
        print('in epoch {} with validation accuracy {}'.format(best_epoch, best_valid_acc))


if __name__ == '__main__':

    evaluate()
