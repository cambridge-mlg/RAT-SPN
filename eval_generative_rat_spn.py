import numpy as np
import pickle
import os
import datasets

result_basefolder = 'results/ratspn/debd/'


def evaluate():
    DEBD = datasets.DEBD
    # if you want to sort according to number of dimensions
    # DEBD = [e[1] for e in sorted([(datasets.DEBD_num_vars[d], d) for d in DEBD])]

    for dataset in DEBD:
        basefolder_dataset = result_basefolder + dataset + '/'
        ls = sorted(os.listdir(basefolder_dataset))

        best_valid_ll = -np.inf
        best_test_ll = -np.inf
        best_epoch = None
        best_model = None
        test_lls = []

        for result_folder in ls:
            argdict = {}
            for a in result_folder.split('__'):
                last_ = a.rfind('_')
                argdict[a[:last_]] = float(a[last_ + 1:])

            try:
                results = pickle.load(open('{}/{}/results.pkl'.format(basefolder_dataset, result_folder), "rb"))
            except:
                print("")
                print("can't load")
                print(result_folder)
                print("")
                continue

            valid_ll = results['valid_LL']
            epoch_idx = np.argmax(valid_ll)
            valid_ll = valid_ll[epoch_idx]
            test_ll = results['test_LL'][epoch_idx]

            test_lls.append(test_ll)

            if valid_ll > best_valid_ll:
                best_valid_ll = valid_ll
                best_test_ll = test_ll
                best_model = result_folder
                best_epoch = epoch_idx

        print("")
        print("Dataset {}".format(dataset))
        print('Test LL: {:.4f}'.format(best_test_ll))
        # print(best_epoch)
        # print(best_model)


if __name__ == '__main__':

    evaluate()




