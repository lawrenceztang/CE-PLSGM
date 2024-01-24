# Overview
Reproduction code for "DIFF2: Differential Private Optimization via Gradient Differences for Nonconvex Distributed Learning"

# Usage
1. Download Gas Turbine, BlogFeedback from UCI machine leanring repository (California Housing will be automatically loaded by sklearn.datasets.fetch_california_housing).
2. For Gas Turbing and BlogFeedback, prepare data/dataset_name/data.pickle that contains the tuple (X, y), where X is the explanatory variables and y is the target variable. Train-test split wille be automatically done in src/utility/load_datasets.py.
   + For Gas Turbing dataset, gt_2011.csv~gt_2015.csv were merged and NOx attribute was removed.
   + For BlogFeedback dataset, only the original train dataset was used. 
3. ```
   conda install --file requirements-conda.txt
   pip3 install --file requirements-pip.txt
   ```
3. `python src/train.py`

Currently, the code is limited on CPU use, but the modification for GPU use is easy. 

# Reproduction of the results in the paper
Run bash scripts/reproduction.sh

# Environment
See requirements.txt