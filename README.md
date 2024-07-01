# DySCo Toolbox

![DySCo Toolbox](https://github.com/Mimbero/DySCo/blob/main/Python/GUI/DySCo%20(3).png)

`DySCo` is a Python and MATLAB toolbox designed to apply the DySCo framework, providing all necessary tools from the beginning to the end of the pipeline.

<p> See: de Alteriis, G., Sherwood, O., Ciaramella, A., Leech, R., Cabral, J., Turkheimer, F,. Expert, P., (in preparation)
<b><i>: DySCo: a general framework for dynamic Functional Connectivity.</i></b></p>

## Table of Contents
- [Background](#background)
- [Overview of the Framework](#overview-of-the-framework)
- [Usage](#usage)
- [Code](#code)
  - [MATLAB](#matlab)
  - [Python](#python)
  - [Pipelines](#pipelines)
  - [GUI](#gui)
- [Installation](#installation)
  - [Dependencies](#dependencies)
- [Help and Support](#help-and-support)
  - [Communication](#communication)
- [Citation](#citation)

## Background

DySCo (Dynamic Symmetric Connectivity) is a mathematically rigorous framework for dynamic Functional Connectivity (dFC). It proposes simple, model-independent metrics to complement existing ones and is applicable to any multivariate time-series data. DySCo leverages the Recurrence Matrix EVD algorithm, which is significantly faster and more memory-efficient than traditional methods. This enables the seamless handling of high-dimensional data, both in time and space, facilitating the real-time exploration of dynamic connectivity patterns.

For detailed information on the mathematical mechanisms of DySCo, refer to the paper by (de Alteriis & Sherwood et al., - in Preparation - [DySCo on BioRxiv](https://www.biorxiv.org/content/10.1101/2024.06.12.598743v1)).

## Overview of the Framework

![DySCo Framework](https://github.com/Mimbero/DySCo/blob/main/Python/GUI/DYSCO_main_12_04.png)

The DySCo framework involves several key steps and methodological decisions:

1. **Data Input and Preprocessing**: Start with raw data and perform appropriate preprocessing.
2. **dFC Matrices**: Choose from four possible dFC matrices (C(t)) as described in the paper.
3. **Subsequent Processing**: Depending on the chosen dFC matrix, perform steps such as window size adjustment or phase extraction to express these matrices into a unified equation.
4. **Eigenvalues and Eigenvectors Calculation**: Use the Recurrence Matrix EVD to calculate eigenvalues and eigenvectors.
5. **DySCo Measures**: Perform dFC analyses and compute the DySCo measures (Norms, Distances, and Entropy).
6. **Derived Measures**: Compute additional measures such as metastability (from norm) and the FCD matrix and reconfiguration speed (from distance).

## Usage

![DySCo Usage](https://github.com/Mimbero/DySCo/blob/main/Python/GUI/DySCO_openerV3.gif)

## Code

### MATLAB (Available Now)

In Matlab we provide the core functions, i.e., the functions to compute the RMEVD, and, from EVD representation, the dysco measures (distance, norm and entropy). 
Note that, as in the Theory, the RMEVD for the cofluctuation matrix is analytical and it is just a z-scoring.
Note that to compute the derived measures, like FCD or metastability, you just need a few lines of code where you call the core functions. 

Example for metastability - being the standard deviation of the norm, would be:
```matlab
meta = std(dysco_norm(eigenvalues, 1));

% Iterate over the time points of interest
FCD(i,j) = dysco_distance(matrix_time_point_i, matrix_time_point_j, what_distance);
```

or for FCD: 
```matlab
% (iterate over the time points of interest)

FCD(i,j) = dysco_distance(matrix_time_point_i,matrix_time_point_j,what_distance)
```

We also provide a short tutorial on how to run each of the core functions in matlab (with associated test data)

### Python
**(N.B. Core functions are released, pipelines will be released on 02.07.24)**

### Installation

1. **Clone the Repository**
    ```sh
    git clone https://github.com/yourusername/dysco.git
    cd dysco
    ```

2. **Create a Virtual Environment**
    It's recommended to create a virtual environment to manage dependencies.
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**
    Install the required dependencies using the `requirements.txt` file.
    ```sh
    pip install -r requirements.txt
    ```
### Core Functions

- **compute_eigenvectors_sliding_corr.py**: Computes eigenvalues and eigenvectors for the given data matrix, using sliding window correlation
- **compute_eigenvectors_sliding_cov.py**: Computes eigenvalues and eigenvectors for the given data matrix, using sliding window covariance
- **compute_eigenvectors_iPA.py**: Computes eigenvalues and eigenvectors for the given data matrix, using instantaneous phase alignment
- **dysco_norm.py**: Computes the norm from the eigenvalues.
- **dysco_distance.py**: Computes the distance between two matrices using the DySCo framework.
- **dysco_mode_alignment.py**: Computes the reconfiguration distance between matrices over time.

For those classy few, there are also python Classes available for these core functions. 

### Pipelines

### GUI

### Help and Support

For further assistance, please contact:

- Oliver Sherwood - Python, General Repo questions: oliver.sherwood@kcl.ac.uk
- Giuseppe de Alteriis - MATLAB: giuseppe.de_alteriis@kcl.ac.uk

## Citation

If you use `DySCo` in a publication, please cite: [DySCo on BioRxiv](https://www.biorxiv.org/content/10.1101/2024.06.12.598743v1)
