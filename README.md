# DySCo

## Background
Dynamic Symmetric Connectivity is a mathematical framework


For full information of the mathematically mechanisms of Dysco find paper by (REF) 


## Installation




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

If the thought of delving into code and inevitable debugging, there is an interface available for DySCo.
You will have to specify the path to your data as well as a subject.txt (with the subject id), and the path to your output folder
(where you want the plots and data_out saved). Additionally, you will have to specify the type of analysis and plotting 
you wish to do. Then press run...and wait. 