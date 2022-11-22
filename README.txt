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

- Create an environment e.g install conda and use
    $ conda create -n 2023_ICLR_ID6100 python=3.6
- Install requirements from requirements file e.g.
    - $ pip install -r requirements.txt
- We run the code (on gpu) with cuda 10.1 gcc 8.3


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
    are the exact scripts to produce our results intable 2


***Other Files***

- The file:
    ./jofsto_code/parameter_fit.py
    estimates DTI,DKI, MSDKI parameters in paper section 4.3
