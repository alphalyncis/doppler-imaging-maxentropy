# !/bin/bash

# Setup virtual environment and install dependencies
virtualenv venv --python=python3.9.6
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt