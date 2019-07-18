# RAT-SPN
Code for UAI'19: Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning

# V0.1
* RAT-SPN model
* Experiments for generative learning using EM

# Quick Start
git clone https://github.com/cambridge-mlg/RAT-SPN

cd RAT-SPN

./install_tensorflow_venv.sh 

source ratspn_venv/bin/activate

python download_preprocess_data.py

python quick_generative_rat_spn.py

python eval_quick_generative_rat_spn.py
