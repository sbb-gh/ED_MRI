***Overview***

TODO changes

This is the submittited code for:
`An Experiment Design Paradigm using Joint Feature Selection and Task Optimization'
Submitted to ICLR 2023 with paper ID6100

Please do not redistribute

This code is an alpha version, will be cleaned and released publicly, on acceptance
This code provides:
    (i) Exact code used for JOFSTO in the paper
    (ii) Exact code used for creating simulations in the paper for table 2
    (iii) Code for parameter fitting



***Environment***

TODO python 3.6* does not install on MAC 
TODO add links to webs

- We tested with python version 3.6.13 and the modules in requirements.txt files
- Installation options:
    - Create conda environment and use: $ conda create -n 2023_ICLR_ID6100 python=3.6.13
    - Install with pip, i.e. install python then $ pip install -r requirements.txt
    - TODO add pyenv version
    - TODO virtual environment
- If GPU is required (recommeneded), code runs on cuda 10.1 gcc 8.3


***Data***

- The files:
    ./simulations/noddi_simulations.py
    ./simulations/verdict_simulations.py
    are exact code to create simulated data used in table 2
- We provide links in the paper to download the two scan datasets (HCP,MUDI)


***Reproducing Results***

- The files:
    ./run_files/JOFSTO_verdict_simulations_papertable2.bash
    ./run_files/JOFSTO_noddi_simulations_papertable2.bash
    are the exact scripts tpyo produce our results intable 2


***Other Files***

- The file:
    ./jofsto_code/parameter_fit.py
    estimates DTI,DKI, MSDKI parameters in paper section 4.3
