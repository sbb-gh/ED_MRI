# EDMRI: A Machine Learning Framework for Experimental Design in Quantitative and Diffusion MRI

EDMRI builds on [TADRED](https://github.com/sbb-gh/tadred) to optimise quantitative and diffusion MRI acquisition protocols for a specified task, such as parameter estimation.

The framework can optimise protocols using either oversampled real MRI data or simulated data. The optimised protocol and trained task network are intended for application to new real MRI acquisitions; simulated application data can also be used for testing and benchmarking.

## Framework overview

This repo builds on the neural-network based algorithm TADRED (TAsk-DRiven Experimental Design in imaging) framework ([code](https://github.com/sbb-gh/tadred), [paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aec2dfc4f5e19acf05c15587c889dbc4-Abstract-Conference.html)). The main additions are a series of wrapper functions that enable quick and easy optimisation of scanning protocols for quantitative and diffusion MRI.

The TADRED framework comprises a **subsampling network** and a **task network**, which are trained jointly during optimisation. The subsampling network identifies the subpart of the data that best supports a task, e.g. parameter mapping, and the task network is trained to undertake that task.


## Schematic
![image](files/tadred-schematic.png)

## Documentation

- **[Installation](docs/installation.md)** — environment setup and dependencies.
- **[Toy example](docs/toy_example.md)** — a complete simulated SANDI example and its output files.
- **[Full workflow walkthrough](docs/full_walkthrough.md)** — detailed protocol optimisation and application using real or simulated training data.
- **[Developer and research notes](docs/developer.md)** — research reproduction and legacy tutorials.

## References

- [TADRED code](https://github.com/sbb-gh/tadred)
- [TADRED paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aec2dfc4f5e19acf05c15587c889dbc4-Abstract-Conference.html)

## Contact

- stefano.blumberg.17@ucl.ac.uk
- slatorp@cardiff.ac.uk
