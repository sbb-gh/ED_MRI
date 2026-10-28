"""Copyright 2024 Stefano B. Blumberg and Paddy J. Slator

Python-based tutorial, this is an overview for the user to understand how to use TADRED on dummy simulated data.
We encourage users to explore different options for data generation, preprocessing
and TADRED hyperparameters - check www.github.com/sbb-gh/tadred/blob/main/tadred/types.py


Overview for cells:
    - Choose data size splits 2
    - Generate dummy data example 3
    - Data format for TADRED 4
    - Option to pass data directly, or save to disk and load 5
    - Option to save output 6
    - TADRED hyperparameters 7,8,9
"""


# %% (1)
# Import modules, README.md for tadred requirements, set global seed

import numpy as np
from tadred import tadred_main, utils

np.random.seed(0)  # Random seed for entire script


# %% (2)
# Data split sizes

n_train = 10**3  # No. training voxels, reduce for faster training speed
n_val = n_train // 10  # No. validations set voxels
n_test = n_train // 10  # No. test set voxels


# %% (3)
# Create data, we provide three options, descriptions below

# Create dummy, randomly generated (positive) data

Cbar = 220  # Num features of densely-sampled data
M = 12  # Number of target regressors
rand = np.random.lognormal  # Random genenerates positive
train_inp, train_tar = rand(size=(n_train, Cbar)), rand(size=(n_train, M))
val_inp, val_tar = rand(size=(n_val, Cbar)), rand(size=(n_val, M))
test_inp, test_tar = rand(size=(n_test, Cbar)), rand(size=(n_test, M))


# %% (4)
# Move data into TADRED format, \bar{C} measurements, M target regresors

data = dict(
    train=train_inp,  # Shape n_train x \bar{C}
    train_tar=train_tar,  # Shape n_train x M
    val=val_inp,  # Shape n_val x \bar{C}
    val_tar=val_tar,  # Shape n_val x M
    test=test_inp,  # Shape n_test x \bar{C}
    test_tar=test_tar,  # Shape n_test x M
)

args = utils.load_base_args()


# %% (5)
# Passing data to TADRED, either directly or save to disk and load

save_to_disk_and_load = False  # set to True to save and then load data

match save_to_disk_and_load:
    case False:
        pass_data = data

    case True:
        # Option to save data to disk, and TADRED load

        data_fil: str = ""  # Add path to saved file
        np.save(data_fil, data)
        print("Saving data as", data_fil)
        pass_data = None
        args.data_norm.data_fil = data_fil


# %% (6)
# Uncomment and fill below to save output
# Output saved as dict in save_fil=<out_base>/<proj_name>/results/<run_name>_all.pkl
# Load with pickle
# args.output.out_base = <ADD>
# args.output.proj_name = <ADD>
# args.output.run_name = <ADD>


# %% (7)
# Simplest version of TADRED, modifying the most important hyperparameters


# Decreasing feature subsets sizes for TADRED to consider
args.tadred_train_eval.feature_set_sizes_Ci = [Cbar, Cbar // 2, Cbar // 4, Cbar // 8, Cbar // 16]

# Feature subset sizess for TADRED evaluated on test data
args.tadred_train_eval.feature_set_sizes_evaluated = [Cbar // 2, Cbar // 4, Cbar // 8, Cbar // 16]

# Scoring net Cbar -> num_units_score[0] -> num_units_score[1] ... -> Cbar units
args.network.num_units_score = [1000, 1000]

# Task net Cbar -> num_units_task[0] -> num_units_task[1] ... -> M units
args.network.num_units_task = [1000, 1000]

tadred_main.run(args, pass_data)


# %% (8)
# Modify less important TADRED hyperparameters, may change results

# Fix score after epoch, E_1 in paper
args.tadred_train_eval.epochs_decay = 25

# Progressively set score to be sample independent across no. epochs, E_2 - E_1 in paper
args.tadred_train_eval.epochs_decay_sigma = 10

# Progressively modify mask across number epochs, E_3 - E_2 in paper
args.tadred_train_eval.epochs_decay = 10

tadred_main.run(args, pass_data)


# %% (9)
# Deep learning training hyperparameters for TADRED inner loop

# Training epochs per step, set large to use early stopping
args.tadred_train_eval.epochs = 10000

# Training learning rate
args.train_pytorch.optimizer_params.lr = 0.0001

# Training batch size
args.train_pytorch.dataloader_params.batch_size = 1500

tadred_main.run(args, pass_data)


print("EOF", __file__)