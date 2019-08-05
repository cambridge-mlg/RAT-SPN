#!/bin/bash

rm -rf ratspn_venv/
virtualenv --system-site-packages -p python3 ./ratspn_venv
source ./ratspn_venv/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade tensorflow-gpu==1.11.0
pip3 install --upgrade sklearn
pip3 install --upgrade filelock
pip3 install --upgrade matplotlib
