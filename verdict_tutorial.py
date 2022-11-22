# (c) Stefano B. Blumberg, do not redistribute or modify

########## (1)
# Import modules, see requirements.txt for jofsto requirements

from simulations import simulations
import numpy as np
from dmipy.data.saved_acquisition_schemes import panagiotaki_verdict_acquisition_scheme
from jofsto_code.jofsto_main import return_argparser, run



########## (2)
# Create data in JOFSTO format

np.random.seed(0) # Random seed for entire script

scheme = panagiotaki_verdict_acquisition_scheme() # Load acquisitions cheme
n_train = 10**4 # No. training voxels
n_val = n_train // 10 # No. validations set voxels
n_test = n_train // 10 # No. test set voxels

# Create train, val, test sets for our example
train_sims = simulations.verdict(n_train, scheme)
val_sims = simulations.verdict(n_val, scheme)
test_sims = simulations.verdict(n_test, scheme)

# Data in JOFSTO format, \bar{C} measurements, M target regresors
data = dict(
    train=train_sims[0], # Shape n_train x \bar{C}
    train_tar=train_sims[1], # Shape n_train x M
    val=val_sims[0], # Shape n_val x \bar{C}
    val_tar=val_sims[1], # Shape n_val x M
    test=test_sims[0], # Shape n_test x \bar{C}
    test_tar=test_sims[1], # Shape n_test x M
)
C_bar = data["train"].shape[1]



########## (3)
# Simplest version of JOFSTO, modifying the most important hyperparameters

jofsto_args = []; parser = return_argparser() # JOFSTO hyperparameters here

# Decreasing feature subsets sizes for JOFSTO to consider
C_i_values = [str(C_bar), "110", "55", "28", "14"]
jofsto_args.extend(["--C_i_values"] + C_i_values)

# Feature subset sizess for JOFSTO evaluated on test data
C_i_eval=["110", "55", "28", "14"]
jofsto_args.extend(["--C_i_eval"] + C_i_eval)

# Scoring net C_bar -> num_units_score[0] -> num_units_score[1] ... -> C_bar units
num_units_score = ["1000", "1000"]
jofsto_args.extend(["--num_units_score"] + num_units_score)

# Task net C_bar -> num_units_task[0] -> num_units_task[1] ... -> M units
num_units_task = ["1000", "1000"]
jofsto_args.extend(["--num_units_task"] + num_units_task)

args = parser.parse_args(jofsto_args)
run(args=args,pass_data=data)



########## (4)
# Modify more JOFSTO hyperparameters, less important, may change results

# Fix score after epoch, E_1 in paper
epochs_fix_sigma = ["25"]
jofsto_args.extend(["--epochs_fix_sigma"] + epochs_fix_sigma)

# Progressively set score to be sample independent across no. epochs, E_2 - E_1 in paper
epochs_decay_sigma = ["10"]
jofsto_args.extend(["--epochs_decay_sigma"] + epochs_decay_sigma)

# Progressively modify mask across number epochs, E_3 - E_2 in paper
epochs_decay = ["10"]
jofsto_args.extend(["--epochs_decay"] + epochs_decay)

args = parser.parse_args(jofsto_args)
run(args=args,pass_data=data)



########## (5)
# Deep learning training hyperparameters for inner loop

# Training epochs per step, set large to trigger early stopping
total_epochs = ["10000"]
jofsto_args.extend(["--total_epochs"] + total_epochs)

# Training learning rate
learning_rate = ["0.0001"]
jofsto_args.extend(["--learning_rate"] + learning_rate)

# Training batch size
batch_size = ["1500"]
jofsto_args.extend(["--batch_size"] + batch_size)

args = parser.parse_args(jofsto_args)
run(args=args,pass_data=data)


########## (6)
# TODO data normalization
#   (i) pre-processing all data
#   (ii) ./utils/calc_affine_norm


print("EOF",__file__)
