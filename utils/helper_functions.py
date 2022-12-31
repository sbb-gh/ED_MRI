import yaml
import numpy as np

def load_yaml(file_path):
    with open(file_path, "r") as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml

def set_simulations_seed(seed):
    np.random.seed(seed)

def jofsto_data_format(train,val,test):
    data_inp = dict(train=train,val=val,test=test)
    data = dict()
    for split in ("train","val","test"):
        data_split = data_inp[split]
        if isinstance(data_split,tuple) and len(data_split) == 2:
            data[split] = data_split[0]
            data[split+"_tar"] = data_split[1]
        else:
            data[split] = data_split

    for key, val in data.items():
        assert isinstance(val, np.ndarray)
    return data
