# An Experiment Design Paradigm using Joint Feature Selection and Task Optimization
# A Machine Learning Framework for Quantitative MRI Protocol Optimisation

This is working code for an upcoming conference paper latest version [link](https://arxiv.org/abs/2210.06891), and an upcoming journal paper.

Please do not distribute.

## Installation Requirements

Code requires: pytorch, numpy, pyyaml.

Some examples require additional packages: notebook (Jupyter Notebook), dmipy (for simulating data).

We use PyTorch v1.7.1, cuda 9.2 on the GPU, we provide example installation instructions below:

### Example Installation using Conda

Conda documentation is [link](https://docs.conda.io).

```bash
$ conda create -n ED_MRI python=3.6.13
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=9.2 pyyaml -c pytorch # GPU installation
# $ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly pyyaml -c pytorch # CPU only
$ conda install notebook; pip install dmipy==1.0.5 # Optional modules
```

### Example Installation using Pyenv

Pyenv documentation is [link](https://github.com/pyenv).

```bash
$ python -m venv ED_MRI_env
$ . ED_MRI_env/bin/activate
$ pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html pyyaml
# $ pip install notebook dmipy==1.0.5  # Optional modules
```

