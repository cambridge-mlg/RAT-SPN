# RAT-SPN
Code for UAI'19: Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning

# V0.2
* RAT-SPN model
* Experiments for generative learning of RAT-SPNs using EM
* Experiments for discriminative learning of RAT-SPNs using Adam

# Setup
git clone https://github.com/cambridge-mlg/RAT-SPN

cd RAT-SPN

./install_tensorflow_venv.sh 

source ratspn_venv/bin/activate

python download_preprocess_data.py

# Quick Run for Generative Experiments
This will simply train a single RAT-SPN (no crossvalidation).

python quick_run_rat_spn_generative.py

python quick_eval_rat_spn_generative.py

# Quick Run for Discriminative Training on MNIST
This will simply train a single RAT-SPN for each depth.

python quick_run_rat_spn_mnist.py

python quick_eval_rat_spn_discriminative.py

# Full Training 
See the run_*.py and eval_*.py files

# Acknowledgments
<img src="https://github.com/cambridge-mlg/RAT-SPN/blob/master/acknowledgement/euc.png"  height="100"/>

This project received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie Grant Agreement No. 797223 (HYBSPN).


