import os
import filelock
import utils
import sys
import subprocess
import time
import json

print("")
print("Discriminative Training of RAT-SPNs on imdb")
print("")

with open('configurations.json') as f:
    configs = json.loads(f.read())

start_time = time.time()
time_limit_seconds = configs['worker_time_limit']
dont_start_if_less_than_seconds = 600.0
base_result_path = "results/ratspn/imdb/"

structure_dict = {}

# depth 1
structure_dict[1] = [
    {'num_recursive_splits': 9, 'num_input_distributions': 10, 'num_sums': 10},
    {'num_recursive_splits': 15, 'num_input_distributions': 15, 'num_sums': 10},
    {'num_recursive_splits': 21, 'num_input_distributions': 20, 'num_sums': 10},
    {'num_recursive_splits': 30, 'num_input_distributions': 26, 'num_sums': 10},
    {'num_recursive_splits': 40, 'num_input_distributions': 36, 'num_sums': 10}]

# depth 2
structure_dict[2] = [
    {'num_recursive_splits': 10, 'num_input_distributions':  8, 'num_sums':  8},
    {'num_recursive_splits': 14, 'num_input_distributions': 14, 'num_sums': 12},
    {'num_recursive_splits': 20, 'num_input_distributions': 20, 'num_sums': 16},
    {'num_recursive_splits': 30, 'num_input_distributions': 26, 'num_sums': 25},
    {'num_recursive_splits': 40, 'num_input_distributions': 38, 'num_sums': 35}]

# depth 3
structure_dict[3] = [
    {'num_recursive_splits': 10, 'num_input_distributions':  8, 'num_sums': 7},
    {'num_recursive_splits': 15, 'num_input_distributions': 14, 'num_sums': 9},
    {'num_recursive_splits': 20, 'num_input_distributions': 18, 'num_sums': 15},
    {'num_recursive_splits': 30, 'num_input_distributions': 23, 'num_sums': 22},
    {'num_recursive_splits': 40, 'num_input_distributions': 35, 'num_sums': 30}]

param_configs = [
        {'dropout_rate_input': 1.0, 'dropout_rate_sums': 1.0},
        {'dropout_rate_input': 1.0, 'dropout_rate_sums': 0.75},
        {'dropout_rate_input': 1.0, 'dropout_rate_sums': 0.5},
        {'dropout_rate_input': 1.0, 'dropout_rate_sums': 0.25},
        {'dropout_rate_input': 0.75, 'dropout_rate_sums': 1.0},
        {'dropout_rate_input': 0.75, 'dropout_rate_sums': 0.75},
        {'dropout_rate_input': 0.75, 'dropout_rate_sums': 0.5},
        {'dropout_rate_input': 0.75, 'dropout_rate_sums': 0.25},
        {'dropout_rate_input': 0.5, 'dropout_rate_sums': 1.0},
        {'dropout_rate_input': 0.5, 'dropout_rate_sums': 0.75},
        {'dropout_rate_input': 0.5, 'dropout_rate_sums': 0.5},
        {'dropout_rate_input': 0.5, 'dropout_rate_sums': 0.25},
        {'dropout_rate_input': 0.25, 'dropout_rate_sums': 1.0},
        {'dropout_rate_input': 0.25, 'dropout_rate_sums': 0.75},
        {'dropout_rate_input': 0.25, 'dropout_rate_sums': 0.5},
        {'dropout_rate_input': 0.25, 'dropout_rate_sums': 0.25}]

num_epochs = 200


def run():
    for split_depth in structure_dict:
        for structure_config in structure_dict[split_depth]:
            for config_dict in param_configs:

                remaining_time = time_limit_seconds - (time.time() - start_time)
                if remaining_time < dont_start_if_less_than_seconds:
                    print("Only {} seconds remaining, stop worker".format(remaining_time))
                    sys.exit(0)

                cmd = "python train_rat_spn.py --store_best_valid_loss --store_best_valid_acc --num_epochs {}".format(num_epochs)
                cmd += " --timeout_seconds {}".format(remaining_time)
                cmd += " --split_depth {}".format(split_depth)

                cmd += " --data_set imdb --data_path data/imdb/"

                for key in sorted(structure_config.keys()):
                    cmd += " --{} {}".format(key, structure_config[key])
                for key in sorted(config_dict.keys()):
                    cmd += " --{} {}".format(key, config_dict[key])

                comb_string = ""
                comb_string += "split_depth_{}".format(split_depth)
                for key in sorted(structure_config.keys()):
                        comb_string += "__{}_{}".format(key, structure_config[key])
                for key in sorted(config_dict.keys()):
                    comb_string += "__{}_{}".format(key, config_dict[key])

                result_path = base_result_path + comb_string
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
