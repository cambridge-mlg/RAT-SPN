import run_rat_spn_generative

run_rat_spn_generative.structure_dict = {}
run_rat_spn_generative.structure_dict[2] = [
    {'num_recursive_splits': 10, 'num_input_distributions':  8, 'num_sums':  8}]
run_rat_spn_generative.base_result_path = "quick_results/ratspn/debd/"
run_rat_spn_generative.num_epochs = 20

run_rat_spn_generative.run()
