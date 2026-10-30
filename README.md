# EDMRI: A Machine Learning Framework for Experimental Design in Quantitative and Diffusion MRI

EDMRI is a machine learning framework for optimising quantitative and diffusion MRI acquisition protocols for a specific task, such as parameter estimation.

The framework produces two outputs: an **optimised acquisition protocol**, designed to efficiently perform the specified task, and a **trained task network**, which performs that task using data acquired with the optimised protocol.

EDMRI requires an **MRI dataset with an extensive set of acquisitions**, which we call the *superdesign*. This dataset can be generated using either real or simulated MRI data.

## Framework overview

EDMRI builds on TADRED (TAsk-DRiven Experimental Design in imaging), a neural-network-based framework for optimising acquisition protocols ([code](https://github.com/sbb-gh/tadred), [paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aec2dfc4f5e19acf05c15587c889dbc4-Abstract-Conference.html)). EDMRI provides wrapper functions that simplify the use of TADRED for quantitative and diffusion MRI.

TADRED comprises two neural networks: a **subsampling network**, which identifies the acquisitions most informative for a specified task, and a **task network**, which learns to perform that task. These networks are trained jointly during optimisation.

## EDMRI workflow

The EDMRI workflow comprises three stages (illustrated below):

1. **Prepare oversampled MRI data:** Define a comprehensive MRI acquisition protocol (the *superdesign*) and generate training data using either real MRI acquisitions or simulations.
2. **Protocol optimisation:** Use TADRED to select an optimised subset of acquisitions and train a task-specific neural network.
3. **Apply optimised protocol and task network
:** Acquire real MRI data using the optimised acquisition protocol and apply the trained network to perform the task.

### Schematic: The three stages of the EDMRI workflow

![The three stages of the EDMRI workflow](files/tadred-schematic.png)

## Documentation

- **[Installation](docs/installation.md)** — environment setup and dependencies.
- **[Getting started](docs/getting_started.md)** — a complete simulated SANDI example and its output files.
- **[Detailed guide](docs/detailed_guide.md)** — detailed protocol optimisation and application using real or simulated training data.
- **[Configuration files](config.md)** - details the YAML configuration files for experimental set up and tracking.
- **[Developer and research notes](docs/developer.md)** — research reproduction and legacy tutorials.

## References

If you use EDMRI in your work please cite the accompanying TADRED paper 

#### 1. Blumberg, S., Slator, P., & Alexander, D. (2024). **Experimental Design for Multi-Channel Imaging via Task-Driven Feature Selection.** *International Conference on Learning Representations (ICLR 2024).* [paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aec2dfc4f5e19acf05c15587c889dbc4-Abstract-Conference.html), [code](https://github.com/sbb-gh/tadred)


## Contact

- stefano.blumberg.17@ucl.ac.uk
- slatorp@cardiff.ac.uk
