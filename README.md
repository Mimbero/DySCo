![Alt text](https://github.com/Mimbero/DySCo/blob/main/Python/GUI/DySCo%20(3).png)

'dysco' is a Python and MATLAB toolbox to apply the DySCo framework. It contains all the necessary tools to apply this
framework from the beginning to end of the pipeline. 
<p> See: de Alteriis, G., Sherwood, O., Ciaramella, A., Leech, R., Cabral, J., Turkheimer, F,. Expert, P., (in preparation) 
</b><i> : DySCo: a general framework for dynamic Functional Connectivity. </i>

## Background

DySCo (Dynamic Symmetric Connectivity) is a unified mathematically rigorous framework for 
dFC approaches that proposes a set of simple, model independent metrics to complement current ones.
It is applicable to any source of data structured as a multivariate time-series.
It is based on the Recurrence Matrix EVD algorithm, a purpose developed new method to compute eigenvectors and eigenvalues,
that outperforms existing methods computational speed and memory requirement by orders of magnitude.
Dysco works seemlessly with very high-dimensional data, both in time and space, and thus paves the road to efficient and systematic
as well as the real-time exploration of dynamic patterns of connectivity.


For full information of the mathematically mechanisms of Dysco find paper by (de Alteriis & Sherwood et al., - in Preparation - https://www.biorxiv.org/content/10.1101/2024.06.12.598743v1) 

### Overview of the framework: 
![Alt text](https://github.com/Mimbero/DySCo/blob/main/Python/GUI/DYSCO_main_12_04.png)

Here we show the main steps involved in the DySCo framework as well as important methodological decisions that must be made when using the framework.
After input of raw data and appropriate pre-processing there are 4 possible dFC
matrices as described in the paper. Based upon the choice of dFC matrix (C(t)),
subsequent processing steps are employed (such as window
size adjustment or extraction of phase) to express these dFC matrices into a
unified equation. We next calculate the eigenvalues and eigenvectors associated with the dFC matrices
using the Recurrence Matrix EVD.
The eigenvalue-eigenvector representation contains all the information needed
to perform dFC analyses, and to compute the DySCo measures.
The three main measures are Norms, Distances, and Entropy.
From these we can obtain derived measures: from the norm it is possible to
compute metastability, from the distance it is possible to compute
the FCD matrix and the reconfiguration speed.


## Usage
![Alt text](https://github.com/Mimbero/DySCo/blob/main/Python/GUI/DySCO_openerV3.gif)

## Code

### MATLAB (Available Now): 
In Matlab we provide the core functions, i.e., the functions to compute the RMEVD, and, from EVD representation, the dysco measures (distance, norm and entropy). 
Note that, as in the Theory, the RMEVD for the cofluctuation matrix is analytical and it is just a z-scoring.
Note that to compute the derived measures, like FCD or metastability, you just need a few lines of code where you call the core functions. 
For example, metastability, being the standard deviation of the norm, would be
meta = std(dysco_norm(eigenvalues,1));

or for FCD: 

% (iterate over the time points of interest)

FCD(i,j) = dysco_distance(matrix_time_point_i,matrix_time_point_j,what_distance)

We also provide a toy example to compute all these quantities on a simulated dataset (which will be uploaded soon, upon paper submission) 

### PYTHON 
### (N.B. Core functions and Pipelines will be released: <i>22.06.24</i>): 

We also provide the core functions in Python as with MATLAB above. To use the core functions please find in 'core_functions' folder. 

### Pipelines

There are a number of pre-built pipelines (only available in python currently) for running the DySCo framework. 
These are currently built to function on fMRI timeseries data (N x t)

### GUI 

#### Watch this space 
<i> Coming Very Soon... </i>

## Installation

### Dependencies required: 

#### Python

<h3><b>Dependencies</b></h3>
<ul style=“list-style-type:disc”>
<li><i>Core Functions:</i></li>
<li>Python</li>
<li>SciPy</li>
<li>tqdm</li>
<li>NumPy</li>

<li><i>Pipelines:</i></li>
<li>Scikit-Learn</li>
<li>Pandas</li>
<li>Matplotlib</li>
<li>Seaborn</li>
<li>Nilearn</li>
<li>random</li>
<li>joblib</li>
<li>threading</li>
</ul>


### Help and Support 

#### Communication
For any further assistance please contact:

<li>oliver.sherwood@kcl.ac.uk - Python, General Repo questions </li>
<li>giuseppe.de_alteriis@kcl.ac.uk - MATLAB </li>

### Citation 

If you use 'DySCo' in a publication please cite: https://www.biorxiv.org/content/10.1101/2024.06.12.598743v1

