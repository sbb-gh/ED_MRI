# An Experiment Design Paradigm using Joint Feature Selection and Task Optimization

This is working code for an upcoming conference paper latest version [link](https://arxiv.org/abs/2210.06891), and also an upcoming journal paper.

Please do not distribute.

## Installation Requirements

Code requires: pytorch, numpy, pyyaml.

Some examples require additional packages: notebook (Jupyter Notebook), dmipy (for simulating data).

The code can be run on CPU or GPU.

We use PyTorch v1.7.1, cuda 9.2 on the GPU. 

We recommend installing in a virtual environment, e.g. using Conda or Pyenv.

Example installations:

### Example Installation with no virtual environment (not recommended)

Install the following packages

```bash
$ pip install numpy torch torchvision torchaudio pyyaml notebook
```

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

## Quick start instructions: optimising an acquisition protocol from simulated data

We anticipant that many code users will want to design an acquisition scheme optimised for their model of choice. This can be done as follows. 

### Step 1. Simulate data for a super-design acquisition scheme using their model 
A "super-design" is a rich acquisition scheme that covers the space of available acquisition parameters. For example to design a diffusion MRI experiment with maximum b-value 3 ms/μm<sup>2</sup>, an appropriate super design might be b=0, 0.01, 0.02, 0.03,..., 2.99, 3 ms/μm<sup>2</sup> with 20 gradient directions at each b-value (20 × 300 = 6000 total measurements).

To simulate data, you can utilise the signal equation of your model of choice (whether diffusion MRI or quantitative MRI). See *simulate\_canonical\_examples.ipynb* and *simulations.py* for examples.

The output from your simulations should be three .npy files: 

* Saved numpy array containing the simulation ground truth parameters (e.g. *model\_name*\_parameters_gt.npy) 
* Saved numpy array containing the simulation super-design signals (e.g. *model\_name*\_signals_super.npy)
* Saved numpy array containing the simulation super-design acquisition parameters (e.g. *model\_name*\_signals_super.npy:)

### Step 2. Train JOFSTO on the simulated data 

The above three files are then used as input for either of:

* *optimise\_protocol\_from\_simulated\_data.ipynb* 
* *optimise\_protocol\_from\_simulated\_data.py*





## Running from the Command Line

```bash
python train_and_eval.py --cfg <YAML_CONFIG_PATH>
```

where <YAML_CONFIG_PATH> is a path to a config file.  See [base config](./base.yaml) for base arguments.

## Tutorial

We provide a more comprehensive tutorial [in tutorial.py](./tutorial.py) that provides examples on generating data, options to load the data into JOFSTO, various hyperparameter choices for JOFSTO, and options to save the results.

## Replicating Results

Duplicating the results for VERDICT and NODDI in table 2 in our [paper](https://arxiv.org/pdf/2210.06891.pdf).  We provide Python code to generate data, train JOFSTO and perform evaulation.  Note, to replicate exact results, we perform a hyperparameter search on the two networks - described in paper-section-B.

### VERDICT Results

```python
import simulations
from jofsto_code import data_processing, jofsto_main, utils
from dmipy.data import saved_acquisition_schemes

scheme = saved_acquisition_schemes.panagiotaki_verdict_acquisition_scheme()
nsamples_train, nsamples_val, nsamples_test  = 10**6, 10**5, 10**5

utils.set_numpy_seed(0)
train_sims = simulations.verdict(nsamples_train, scheme)
val_sims = simulations.verdict(nsamples_val, scheme)
test_sims = simulations.verdict(nsamples_test, scheme)

data = data_processing.jofsto_data_format(train_sims,val_sims,test_sims)
args = utils.load_yaml("./base.yaml")

args["jofsto_train_eval"]["C_i_values"] = [220, 110, 55, 28, 14]
args["jofsto_train_eval"]["C_i_eval"] = [110, 55, 28, 14]

# Set the below to network sizes, see paper-section-B
args["network"]["num_units_score"] = [] # CHANGE e.g. [1000, 1000]
args["network"]["num_units_task"] = [] # CHANGE e.g. [1000, 1000]

jofsto_main.run(args, data)
```

### NODDI Results

```python
import simulations
from jofsto_code import data_processing, jofsto_main, utils
from dmipy.data import saved_acquisition_schemes

scheme = saved_acquisition_schemes.isbi2015_white_matter_challenge_scheme()
nsamples_train, nsamples_val, nsamples_test  = 10**5, 10**4, 10**4

utils.set_numpy_seed(0)
train_sims = simulations.noddi(nsamples_train, scheme)
val_sims = simulations.noddi(nsamples_val, scheme)
test_sims = simulations.noddi(nsamples_test, scheme)

data = data_processing.jofsto_data_format(train_sims,val_sims,test_sims)
args = utils.load_yaml("./base.yaml")

args["jofsto_train_eval"]["C_i_values"] = [3612, 1806, 903, 452, 226]
args["jofsto_train_eval"]["C_i_eval"] = [1806, 903, 452, 226]

# Set the below to network sizes, see paper-section-B
args["network"]["num_units_score"] = [] # CHANGE e.g. [1000, 1000]
args["network"]["num_units_task"] = [] # CHANGE e.g. [1000, 1000]

jofsto_main.run(args, data)
```
