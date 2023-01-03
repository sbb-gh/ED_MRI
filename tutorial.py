"""
Python-based tutorial for https://arxiv.org/pdf/2210.06891.pdf

We encourage users to explore different options for
data generation, preprocessing and JOFSTO hyperparameters.

TODO Code runs on gpu automatically, detection torch.cuda_is_available

Overview for cells:
    - Choose data size splits 2
    - Generate data examples 3-A/B/C
    - Data format for JOFSTO 4
    - Option to pass data directly, or save to disk and load 5-A/B
    - Option to save output 6
    - JOFSTO hyperparameters 7,8,9 in order of importance.
    - Data normalization 10
"""


########## (1)
# Import modules, see requirements.txt for jofsto requirements, set global seed

import numpy as np
from jofsto_code import jofsto_main, utils

np.random.seed(0)  # Random seed for entire script


########## (2)
# Data split sizes

n_train = 10**3  # No. training voxels, reduce for faster training speed
n_val = n_train // 10  # No. validations set voxels
n_test = n_train // 10  # No. test set voxels


########## (3-A)
# Create dummy, randomly generated (positive) data

C_bar = 220
M = 12  # Number of input measurements \bar{C}, Target regressors
rand = np.random.lognormal  # Random genenerates positive
train_inp, train_tar = rand(size=(n_train, C_bar)), rand(size=(n_train, M))
val_inp, val_tar = rand(size=(n_val, C_bar)), rand(size=(n_val, M))
test_inp, test_tar = rand(size=(n_test, C_bar)), rand(size=(n_test, M))


########## (3-B)
# Generate data using VERDICT model and scheme [link](https://pubmed.ncbi.nlm.nih.gov/25426656/)
# Requires python package dmipy [link](https://github.com/AthenaEPI/dmipy) tested on v1.0.5

import simulations
from dmipy.data.saved_acquisition_schemes import panagiotaki_verdict_acquisition_scheme

# Create train, val, test sets for our example from a scheme
scheme = panagiotaki_verdict_acquisition_scheme()  # Load acquisitions cheme
train_inp, train_tar = simulations.verdict(n_train, scheme)
val_inp, val_tar = simulations.verdict(n_val, scheme)
test_inp, test_tar = simulations.verdict(n_test, scheme)
C_bar = train_inp.shape[1]
M = train_tar.shape[1]  # C_bar,M same for val, test data


########## (3-C)
# Generate data using NODDI model [link](https://pubmed.ncbi.nlm.nih.gov/22484410/)
# Uses acquisition scheme [link](https://pubmed.ncbi.nlm.nih.gov/28643354/)
# Requires python package dmipy [link](https://github.com/AthenaEPI/dmipy) tested on v1.0.5

import simulations
from dmipy.data.saved_acquisition_schemes import isbi2015_white_matter_challenge_scheme

# Create train, val, test sets for our example from a scheme
scheme = isbi2015_white_matter_challenge_scheme()  # Load acquisitions cheme
train_inp, train_tar = simulations.noddi(n_train, scheme)
val_inp, val_tar = simulations.noddi(n_val, scheme)
test_inp, test_tar = simulations.noddi(n_test, scheme)
C_bar = train_inp.shape[1]
M = train_tar.shape[1]  # C_bar,M same for val, test data


########## (4)
# Load data into JOFSTO format

# Data in JOFSTO format, \bar{C} measurements, M target regresors
data = dict(
    train=train_inp,  # Shape n_train x \bar{C}
    train_tar=train_tar,  # Shape n_train x M
    val=val_inp,  # Shape n_val x \bar{C}
    val_tar=val_tar,  # Shape n_val x M
    test=test_inp,  # Shape n_test x \bar{C}
    test_tar=test_tar,  # Shape n_test x M
)

# Load base JOFSTO hyperparameters
args = utils.load_yaml("./base.yaml")

"""
########## (5-A)
# Option to save data to disk, and JOFSTO load

data_fil = ""  # Add path to saved file
np.save(data_fil, data)
print("Saving data as", data_fil)
pass_data = None
args["data_norm"]["data_fil"] = data_fil
"""


########## (5-B)
# Option to pass data to JOFSTO directly

pass_data = data


########## (6)
# Option to save the output
"""
# Output saved as dict in save_fil=<out_base>/<proj_name>/results/<run_name>_all.npy
# Load with np.load(str(save_fil),allow_pickle=True).item()
args["output"]["out_base"] = <ADD>
args["output"]["proj_name"] = <ADD>
args["output"]["run_name"] = <ADD>
"""

########## (7)
# Simplest version of JOFSTO, modifying the most important hyperparameters


# Decreasing feature subsets sizes for JOFSTO to consider
args["C_i_values"] = [C_bar, C_bar // 2, C_bar // 4, C_bar // 8, C_bar // 16]

# Feature subset sizess for JOFSTO evaluated on test data
args["C_i_eval"] = [C_bar // 2, C_bar // 4, C_bar // 8, C_bar // 16]

# Scoring net C_bar -> num_units_score[0] -> num_units_score[1] ... -> C_bar units
args["network"]["num_units_score"] = [1000, 1000]

# Task net C_bar -> num_units_task[0] -> num_units_task[1] ... -> M units
args["network"]["num_units_task"] = [1000, 1000]

jofsto_main.run(args, pass_data)


########## (8)
# Modify more JOFSTO hyperparameters, less important, may change results

# Fix score after epoch, E_1 in paper
args["epochs_fix_sigma"] = 25

# Progressively set score to be sample independent across no. epochs, E_2 - E_1 in paper
args["epochs_decay_sigma"] = 10

# Progressively modify mask across number epochs, E_3 - E_2 in paper
args["epochs_decay"] = 10

jofsto_main.run(args, pass_data)


########## (9)
# Deep learning training hyperparameters for inner loop

# Training epochs per step, set large to use early stopping
args["total_epochs"] = 10000

# Training learning rate
args["learning_rate"] = 0.0001

# Training batch size
args["batch_size"] = 1500

jofsto_main.run(args, pass_data)


########## (10)
# TODO data normalization
#   (i) pre-processing all data
#   (ii) ./utils/calc_affine_norm


print("EOF", __file__)
