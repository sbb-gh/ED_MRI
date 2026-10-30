# Installation

[README](../README.md) · [Getting started](getting_started.md) · [Detailed guide](detailed_guide.md) . [Configuration files](config.md)

This page covers environment setup and installation.

> **Note:** The installation and version details below are retained from the existing README and may need checking against the current package configuration.

## Installation

We recommend installing EDMRI and its dependencies in a **Python virtual environment** to keep them isolated from other Python installations and packages.

The instructions below assume a top-level directory, `<INSTALL_DIR>`, containing the virtual environment, EDMRI and TADRED installations. This directory structure is recommended because it is assumed by some of the example notebooks.

Although EDMRI has been successfully tested with several Python versions, we recommend **Python 3.10.4**. We also recommend [pyenv](https://github.com/pyenv/pyenv) for managing Python versions.

The code has been fully tested with **PyTorch 2.0.0**, **CUDA 11.7** (GPU), **DIPY 1.5.0**, **NiBabel 5.1.0** and **dmipy 1.0.5**.


### Installation Part 1: Directory set up
First create and move to the desired installation directory.

```
mkdir <INSTALL_DIR>
cd <INSTALL_DIR>
```

### Installation Part 2: ED_MRI repo
Clone this ED_MRI repo to ```<INSTALL_DIR>``` as follows.

```bash
git clone https://github.com/sbb-gh/ED_MRI.git
```

### Installation Part 3: TADRED repo
Now clone the relevant fork of the TADRED code to ```<INSTALL_DIR>``` as follows.

```bash
git clone https://github.com/PaddySlator/tadred.git
```

The relevant fork of the TADRED github repo is located [here](https://github.com/PaddySlator/tadred/tree/main).

<!--TODO: You can also use pip install (doesn't work on mac)

```bash
pip install git+https://github.com/sbb-gh/tadred.git@main
```-->


### Installation Part 4: Environment

Now create an activate a virtual environment in ```<INSTALL_DIR>```.  We provide two examples either using venv or Conda:

#### Option 1: Install Environment Using python venv module
The venv documentation is [here](https://docs.python.org/3/library/venv.html).

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Option 2: Install Environment Using Conda
Conda documentation is [here](https://docs.conda.io/en/latest/).

```bash
conda create -n ED_MRI_env python=3.10.4
conda activate ED_MRI_env
```


### Installation Part 5: TADRED and Dependencies

Now install TADRED and the dependencies using pip.

After first updating pip, install EDMRI

```bash
pip install --upgrade pip
cd ED_MRI
pip install -e .
```


and TADRED

```bash
cd ../tadred
pip install -e .
```

and then the additional dependencies

```bash
pip install notebook matplotlib scipy nibabel
```


<!--#### TADRED dependencies

TADRED requires pytorch, numpy, pyyaml, and hydra which can be installed as follows:

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install pyyaml hydra-core==1.3
pip install numpy
```

#### ED_MRI dependencies
Required to run the example jupyter notebooks:

```bash
pip install notebook
```-->

#### [Optional] Full dependencies for replication of results
Optional modules to generate the data and run a full replication of the results: dipy, dmipy, nibabel. Note that dmipy is no longer supported and can often be difficult to install. If you are struggling try cloning this repo https://github.com/PaddySlator/dmipy into ```<INSTALL_DIR>``` instead.


```bash
pip install git+https://github.com/AthenaEPI/dmipy.git@1.0.1
pip install dipy==1.9.0
pip install nibabel==5.1.0
```


<!--```bash
pip install numpy==1.23.4 git+https://github.com/AthenaEPI/dmipy.git@1.0.1
pip install dipy==1.9.0
pip install nibabel==5.1.0
pip install git+https://github.com/sbb-gh/tadred.git@main # can also install tadred from source to examine/modify files from www.github.com/sbb-gh/tadred
pip install notebook
```-->


<!--We provide two options for installing the code:-->

<!--#### Python Package from Source

```bash
pip install git+https://github.com/sbb-gh/ED_MRI.git@main notebook
```-->


<!--```bash
pip install numpy==1.23.4 git+https://github.com/AthenaEPI/dmipy.git@1.0.1
pip install dipy==1.9.0
pip install nibabel==5.1.0
pip install git+https://github.com/sbb-gh/tadred.git@main # can also install tadred from source to examine/modify files from www.github.com/sbb-gh/tadred
pip install notebook
```-->

