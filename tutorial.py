"""
(c) Stefano B. Blumberg, do not redistribute or modify

Python-based tutorial for <Add paper link>

Overview for cells:
    - Choose data size splits 2
    - Generate data examples 3-A/B/C
    - Data format for JOFSTO 4
    - Option to pass data directly, or save to disk and load 5-A/B
    - JOFSTO hyperparameters 6,7,8
    - Data normalization 9
"""


########## (1)
# Import modules, see requirements.txt for jofsto requirements, set global seed

import numpy as np
from jofsto_code.jofsto_main import return_argparser, run

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

from utils import simulations
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

from utils import simulations
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

jofsto_args = []
parser = return_argparser()  # JOFSTO hyperparameters here


########## (5-A)
# Option to save data to disk, and JOFSTO load

data_fil = ""  # Add path to saved file
np.save(data_fil, data)
print("Saving data as", data_fil)
pass_data = None
jofsto_args.extend(["--data_fil", data_fil])


########## (5-B)
# Option to pass data to JOFSTO directly

pass_data = data


########## (6)
# Simplest version of JOFSTO, modifying the most important hyperparameters


# Decreasing feature subsets sizes for JOFSTO to consider
C_i_values = [C_bar, C_bar // 2, C_bar // 4, C_bar // 8, C_bar // 16]
jofsto_args.extend(["--C_i_values"] + [str(val) for val in C_i_values])

# Feature subset sizess for JOFSTO evaluated on test data
C_i_eval = [C_bar // 2, C_bar // 4, C_bar // 8, C_bar // 16]
jofsto_args.extend(["--C_i_eval"] + [str(val) for val in C_i_eval])

# Scoring net C_bar -> num_units_score[0] -> num_units_score[1] ... -> C_bar units
num_units_score = [1000, 1000]
jofsto_args.extend(["--num_units_score"] + [str(val) for val in num_units_score])

# Task net C_bar -> num_units_task[0] -> num_units_task[1] ... -> M units
num_units_task = [1000, 1000]
jofsto_args.extend(["--num_units_task"] + [str(val) for val in num_units_task])

args = parser.parse_args(jofsto_args)
run(args=args, pass_data=pass_data)


########## (7)
# Modify more JOFSTO hyperparameters, less important, may change results

# Fix score after epoch, E_1 in paper
epochs_fix_sigma = 25
jofsto_args.extend(["--epochs_fix_sigma", str(epochs_fix_sigma)])

# Progressively set score to be sample independent across no. epochs, E_2 - E_1 in paper
epochs_decay_sigma = 10
jofsto_args.extend(["--epochs_decay_sigma", str(epochs_decay_sigma)])

# Progressively modify mask across number epochs, E_3 - E_2 in paper
epochs_decay = 10
jofsto_args.extend(["--epochs_decay", str(epochs_decay)])

args = parser.parse_args(jofsto_args)
run(args=args, pass_data=pass_data)


########## (8)
# Deep learning training hyperparameters for inner loop

# Training epochs per step, set large to trigger early stopping
total_epochs = 10000
jofsto_args.extend(["--total_epochs", str(total_epochs)])

# Training learning rate
learning_rate = 0.0001
jofsto_args.extend(["--learning_rate", str(learning_rate)])

# Training batch size
batch_size = 1500
jofsto_args.extend(["--batch_size", str(batch_size)])

args = parser.parse_args(jofsto_args)
run(args=args, pass_data=pass_data)


########## (9)
# TODO data normalization
#   (i) pre-processing all data
#   (ii) ./utils/calc_affine_norm


print("EOF", __file__)
