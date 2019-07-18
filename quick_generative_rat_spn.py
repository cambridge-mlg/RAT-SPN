import generative_rat_spn

generative_rat_spn.structure_dict = {}
generative_rat_spn.structure_dict[2] = [
    {'num_recursive_splits': 10, 'num_input_distributions':  8, 'num_sums':  8}]
generative_rat_spn.base_result_path = "quick_results/ratspn/debd/"
generative_rat_spn.num_epochs = 20

generative_rat_spn.run()
