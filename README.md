# Experimental Design Journal Paper

This is working code for an upcoming conference paper latest version [link](https://arxiv.org/abs/2210.06891), and also an upcoming journal paper.

The neural-network based algorithm TADRED is available [here](https://github.com/sbb-gh/tadred).

If you find this repo useful, please consider citing the paper:

## Contact

stefano.blumberg.17@ucl.ac.uk

slatorp@cardiff.ac.uk


## Installation Part 1: Environment

First create an environment and enter it, we use Python v3.10.4.  We provide two examples either using Pyenv or Conda:

## Pyenv

```bash
# Pyenv documentation is [link](https://github.com/pyenv), where <INSTALL_DIR> is the directory the virtual environment is installed in.
python3.10 -m venv <INSTALL_DIR>/ED_MRI_env # Use compatible Python version e.g. 3.10.4
. <INSTALL_DIR>/ED_MRI_env/bin/activate
```

## Conda

```bash
# Conda documentation is [link](https://docs.conda.io/en/latest/), where <INSTALL_DIR> is the directory the virtual environment is installed in.
conda create -n ED_MRI_env python=3.10.4
conda activate ED_MRI_env
```


## Installation Part 2: TADRED and other Packages

Code requires:<br>
[tadred](https://github.com/sbb-gh/tadred/tree/main): the novel method presented in the paper with dependencies pytorch, numpy, pyyaml, hydra,<br>
optional modules to generate the data: dipy, dmipy, nibabel.<br>
<br>
Code is tested using PyTorch v2.0.0, cuda 11.7 on the GPU, dipy v1.5.0, nibabel v5.1.0, dmipy v1.0.5.

We provide two options for installing the code:

### Python Package from Source


```bash
pip install git+https://github.com/sbb-gh/ED_MRI.git@main notebook
```

### Using pip

```bash
pip install numpy==1.23.4 git+https://github.com/AthenaEPI/dmipy.git@1.0.1
pip install dipy==1.9.0
pip install nibabel==5.1.0
pip install git+https://github.com/sbb-gh/tadred.git@main # can also install tadred from source: www.github.com/sbb-gh/tadred
pip install notebook
```


## Quick start instructions: optimising an acquisition protocol from simulated data

We anticipant that many code users will want to design an acquisition scheme optimised for their model of choice. This can be done as follows.

### Step 1. Simulate data for a super-design acquisition scheme using their model
A "super-design" is a rich acquisition scheme that covers the space of available acquisition parameters. For example to design a diffusion MRI experiment with maximum b-value 3 ms/μm<sup>2</sup>, an appropriate super design might be b=0, 0.01, 0.02, 0.03,..., 2.99, 3 ms/μm<sup>2</sup> with 20 gradient directions at each b-value (20 × 300 = 6000 total measurements).

To simulate data, you can utilise the signal equation of your model of choice (whether diffusion MRI or quantitative MRI). See *simulate\_canonical\_examples.ipynb* and *simulations.py* for examples.

The output from your simulations should be three .npy files:

* *model\_name*\_parameters_gt.npy: array containing the simulation ground truth parameters
* *model\_name*\_signals_super.npy: array containing the simulation super-design signals
* *model\_name*\_signals_super.npy: array containing the simulation super-design acquisition parameters

### Step 2. Train TADRED on the simulated data

The above three files are then used as input for either of:

* *optimise\_protocol\_from\_simulated\_data.ipynb*
* *optimise\_protocol\_from\_simulated\_data.py*

## Running from the Command Line

```bash
python train_and_eval.py --cfg <YAML_CONFIG_PATH>
```

where <YAML_CONFIG_PATH> is a path to a config file.  See [base config](./base.yaml) for base arguments.

## Tutorial

We provide a more comprehensive tutorial [in tutorial.py](./tutorial.py) that provides examples on generating data, options to load the data into TADRED, various hyperparameter choices for TADRED, and options to save the results.

