import scipy.io
import numpy as np
load_dir = "/home/blumberg/Bureau/z_Automated_Measurement/Data/FLEXI/"
save_dir = "/home/blumberg/Bureau/z_Automated_Measurement/Data/FLEXI_Processed/"


data = dict()
for file_n in [
    "train_inp",
    "val_inp",
    "test_inp",
    "train_tar",
    "val_tar",
    "test_tar",
    ]:
    load_f = load_dir + file_n
    data[file_n] = scipy.io.loadmat(load_f)[file_n]

# TODO This doesn't work
file_n = "measurement_to_dependents"
load_f = load_dir + file_n
data[file_n] = scipy.io.loadmat(load_f)['__function_workspace__']

np.save(save_dir+"FLEXI",data)

print("EOF",__file__)
