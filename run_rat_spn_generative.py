import os
import filelock
import datasets
import sys
import subprocess
import time
import utils
import json

print("")
print("Generative Training of RAT-SPNs on 20 binary datasets")
print("")

with open('configurations.json') as f:
    configs = json.loads(f.read())

start_time = time.time()
time_limit_seconds = configs['worker_time_limit']
dont_start_if_less_than_seconds = 600.0

optimizer = "em"

if optimizer == "em":
    base_result_path = "results/ratspn/debd/"
elif optimizer == "adam":
    base_result_path = "results/ratspn/debd_adam/"
else:
    raise AssertionError("unknown optimizer")

structure_dict = {}

# depth 1
structure_dict[1] = [
    {'num_recursive_splits': 10, 'num_input_distributions': 10, 'num_sums': 10},
    {'num_recursive_splits': 25, 'num_input_distributions': 20, 'num_sums': 10},
    {'num_recursive_splits': 50, 'num_input_distributions': 45, 'num_sums': 10}]

# depth 2
structure_dict[2] = [
    {'num_recursive_splits':  4, 'num_input_distributions':  5, 'num_sums':  5},
    {'num_recursive_splits': 10, 'num_input_distributions':  8, 'num_sums':  8},
    {'num_recursive_splits': 15, 'num_input_distributions': 15, 'num_sums': 15}]

# depth 3
structure_dict[3] = [
    {'num_recursive_splits':  3, 'num_input_distributions':  5, 'num_sums':  3},
    {'num_recursive_splits': 10, 'num_input_distributions':  6, 'num_sums':  5},
    {'num_recursive_splits': 16, 'num_input_distributions': 10, 'num_sums': 10}]

# depth 4
structure_dict[4] = [
    {'num_recursive_splits':  3, 'num_input_distributions':  3, 'num_sums':  3},
    {'num_recursive_splits':  6, 'num_input_distributions':  5, 'num_sums':  5},
    {'num_recursive_splits': 10, 'num_input_distributions': 10, 'num_sums':  8}]

num_epochs = 100


def run():
    for dataset in datasets.DEBD:
        for split_depth in structure_dict:
            for structure_config in structure_dict[split_depth]:
                remaining_time = time_limit_seconds - (time.time() - start_time)
                if remaining_time < dont_start_if_less_than_seconds:
                    print("Only {} seconds remaining, stop worker".format(remaining_time))
                    sys.exit(0)

                cmd = "python train_rat_spn.py --store_best_valid_loss --store_model_max 1 --num_epochs {}".format(num_epochs)
                cmd += " --discrete_leaves --lambda_discriminative 0.0"
                cmd += " --optimizer " + optimizer
                if dataset == "ad" or optimizer == "adam":
                    cmd += " --batch_size 100"
                else:
                    cmd += " --batch_size 200"
                cmd += " --timeout_seconds {}".format(remaining_time)
                cmd += " --split_depth {}".format(split_depth)
                cmd += " --data_path data/DEBD"
                cmd += " --data_set {}".format(dataset)

                for key in sorted(structure_config.keys()):
                    cmd += " --{} {}".format(key, structure_config[key])

                comb_string = ""
                comb_string += "split_depth_{}".format(split_depth)
                for key in sorted(structure_config.keys()):
                    comb_string += "__{}_{}".format(key, structure_config[key])

                result_path = base_result_path + dataset + '/' + comb_string
                cmd += " --result_path " + result_path

                ###
                print(cmd)

                utils.mkdir_p(result_path)
                lock_file = result_path + "/file.lock"
                done_file = result_path + "/file.done"
                lock = filelock.FileLock(lock_file)
                try:
                    lock.acquire(timeout=0.1)
                    if os.path.isfile(done_file):
                        print("   already done -> skip")
                    else:
                        sys.stdout.flush()
                        ret_val = subprocess.call(cmd, shell=True)
                        if ret_val == 7:
                            lock.release()
                            print("Task timed out, stop worker")
                            sys.exit(0)
                        os.system("touch {}".format(done_file))
                    lock.release()
                except filelock.Timeout:
                    print("   locked -> skip")

if __name__ == '__main__':
    run()
