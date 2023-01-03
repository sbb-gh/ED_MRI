# An Experiment Design Paradigm using Joint Feature Selection and Task Optimization

This is working code for an upcoming conference paper latest version [link](https://arxiv.org/abs/2210.06891), and also an upcoming journal paper.

Please do not distribute.

## Installation Requirements

Code requires: pytorch, numpy, pyyaml.

Some examples require additional packages: notebook (Jupyter Notebook), dmipy (for simulating data).

We use PyTorch v1.7.1, cuda 9.2 on the GPU, and provide example installation instructions:

### Example Installation using Conda

Conda documentation is [link](https://docs.conda.io).

```bash
$ conda create -n ED_MRI python=3.6.13
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=9.2 pyyaml -c pytorch # GPU installation
# $ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly pyyaml -c pytorch # CPU only
$ conda install notebook; pip install dmipy==1.0.5 # Optional modules
```

### Example Installation using Pyenv

Pyenv documentation is [link](https://github.com/pyenv), where <INSTALL_DIR> is the directory the virtual environment is installed in.

```bash
$ python -m venv <INSTALL_DIR>/ED_MRI_env # Assume using compatible Python version e.g. 3.6.13
$ . <INSTALL_DIR>/ED_MRI_env/bin/activate
$ pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html pyyaml
# $ pip install notebook dmipy==1.0.5  # Optional modules
```

## Running from the Command Line

```bash
python train_and_eval.py --cfg <YAML_CONFIG_PATH>
```

where <YAML_CONFIG_PATH> is a path to a config file.  See [base config](./base.yaml) for base arguments.


## Tutorial

We provide a tutorial [in tutorial.py](./tutorial.py) that provides examples on generating data, options to load the data into JOFSTO, various hyperparameter choices for JOFSTO, and options to save the results.

## Results

Duplicating the results for VERDICT and NODDI in table 2 in our [paper](https://arxiv.org/pdf/2210.06891.pdf).  We provide Python code to generate data, train JOFSTO and perform evaulation.  Note, to replicate exact results, we perform a hyperparameter search on the two networks - described in paper-section-B.

### VERDICT Results

```python
import simulations
from jofsto_code import jofsto_main, utils
from dmipy.data import saved_acquisition_schemes

scheme = saved_acquisition_schemes.panagiotaki_verdict_acquisition_scheme()
nsamples_train, nsamples_val, nsamples_test  = 10**6, 10**5, 10**5

utils.set_numpy_seed(0)
train_sims = simulations.verdict(nsamples_train, scheme)
val_sims = simulations.verdict(nsamples_val, scheme)
test_sims = simulations.verdict(nsamples_test, scheme)

data = utils.jofsto_data_format(train_sims,val_sims,test_sims)
args = utils.load_yaml("./base.yaml")

args["C_i_values"] = [220, 110, 55, 28, 14]
args["C_i_eval"] = [110, 55, 28, 14]

# Set the below to network sizes, see paper-section-B
args["network"]["num_units_score"] = [] # CHANGE e.g. [1000, 1000]
args["network"]["num_units_task"] = [] # CHANGE e.g. [1000, 1000]

jofsto_main.run(args, data)
```

### NODDI Results

```python
import simulations
from jofsto_code import jofsto_main, utils
from dmipy.data import saved_acquisition_schemes

scheme = saved_acquisition_schemes.isbi2015_white_matter_challenge_scheme()
nsamples_train, nsamples_val, nsamples_test  = 10**5, 10**4, 10**4

utils.set_numpy_seed(0)
train_sims = simulations.noddi(nsamples_train, scheme)
val_sims = simulations.noddi(nsamples_val, scheme)
test_sims = simulations.noddi(nsamples_test, scheme)

data = utils.jofsto_data_format(train_sims,val_sims,test_sims)
args = utils.load_yaml("./base.yaml")

args["C_i_values"] = [3612, 1806, 903, 452, 226]
args["C_i_eval"] = [1806, 903, 452, 226]

# Set the below to network sizes, see paper-section-B
args["network"]["num_units_score"] = [] # CHANGE e.g. [1000, 1000]
args["network"]["num_units_task"] = [] # CHANGE e.g. [1000, 1000]

jofsto_main.run(args, data)
```
