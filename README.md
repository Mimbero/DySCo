![Alt text](https://github.com/Mimbero/DySCo/blob/main/Python/GUI/DySCo%20(3).png)

## Background

DySCo (Dynamic Symmetric Connectivity) is a unified mathematically rigorous framework for 
dFC approaches that proposes a set of simple, model independent metrics to complement current ones.
It is applicable to any source of data structured as a multivariate time-series.
It is based on the Recurrence Matrix EVD algorithm, a purpose developed new method to compute eigenvectors and eigenvalues,
that outperforms existing methods computational speed and memory requirement by orders of magnitude.
Dysco works seemlessly with very high-dimensional data, both in time and space, and thus paves the road to efficient and systematic
as well as the real-time exploration of dynamic patterns of connectivity.


For full information of the mathematically mechanisms of Dysco find paper by (REF) 

![Alt text]()


![Alt text](https://github.com/Mimbero/DySCo/blob/main/Python/GUI/DySCO_openerV3.gif)
## Installation

### Dependencies required: 

#### Python

Core_functions:
- scipy
- tqdm
- numpy

Pipeline:
- nibabel
- random
- numpy
- matplotlib
- sys
- os
- tqdm
- joblib
- threading
- warnings
- scipy





## Usage 

### Core Functions

To use the core functions please find in 'core_functions' folder, available for either MATLAB or Python. 
As a brief break down of the utility of each function. 

- compute_eigs = 
- compute_norm = 
- eida_distance = 
- eida_reconf_distance =

### Pipelines

There are a number of pre-built pipelines (only available in python currently) for running the DySCo framework. 
These are currently built to function on fMRI timeseries data (N x t)

Main version (29.04) = 'Simple_Pipe.py'

### GUI 

If the thought of delving into code and inevitable debugging causes distress, there is an interface available for DySCo.
You will have to specify the path to your data as well as a subject.txt (with the subject id), and the path to your output folder
(where you want the plots and data_out saved). Additionally, you will have to specify the type of analysis and plotting 
you wish to do. Then press run...and wait. 
