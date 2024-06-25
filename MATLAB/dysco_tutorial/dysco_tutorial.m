% if you want to learn how to use dysco this is the right place! This
% scripts teaches you how to run the core functions to build your dysco
% analysis pipeline

clear,close
addpath("../core_functions");

%% 1 load the timeseries. This should be a matrix, each row is a timepoint,
% each column is a signal/brain area/feature. So it's TxN

load("example_fmri.mat");


%% 2 after you have selected the type of matrix (see paper) and preprocessed
% the data, run the recurrence matrix evd for the specified matrix. For
% example here we are running it for a sliding window correlation matrix
% with a window of 21 (odd numbers for symmetry). Remember that the rank (=n of non null eigenvalues)
% is lower than window size (see paper). In this case we calculate the
% first 10 eigenvectors as an example

half_window_size = 10;
n_eigen = 10;

[eigenvectors,eigenvalues] = compute_eigenvectors_sliding_corr(example_fmri,half_window_size,n_eigen);

% ok now you have eigenvectors and eigenvalues. Eigenvalues is a 2D matrix,
% where each column is our 10 eigenvalues at each time point. Eigenvectors
% is 3D, because it is a matrix of eigenvectors for each time point. Every
% column of the matrix is an eigenvector, and indeed every matrix has 10
% column


%% 3 compute DySCo measures: now that we have this EVD representation of our
% sliding - window correlation matrix, we can compute the DySCo measures: 


%% a. NORM - this is the time-varying norm, computed from eigenvalues (see
% paper), so at each time point you have the norm of the matrix. Let us
% compute the norm 2, but there are different norms available (see paper)

norm = dysco_norm(eigenvalues,2);

% a1. from norm we can compute a derived measure, which is spectral
% metastability - see paper

metastability = std(norm); 

%% b. DISTANCE - we can compute the distance between dynamic matrices at 2
% different time points. For example, let us use the distance 2 to compute
% the Functional Connectivity Dynamics (FCD) matrix 

T = size(eigenvalues,2);
FCD = zeros(T,T);

for i = 1:T
    for j=i+1:T
        FCD(i,j) = dysco_distance(eigenvectors(:,:,i),eigenvectors(:,:,j),2);
        FCD(j,i) = FCD(i,j);
    end
end


% reconfiguration speed is just the distance between the matrix at time t
% and the matrix at time t-lag, so if we already have the FCD matrix, the
% reconfiguration speed will be just derived from that: (here we suppose
% lag = 5)

lag = 5;
speed = zeros(1,T-lag);
for i=1:T-lag
    speed(i) = FCD(i,i+lag);
end


%% c. ENTROPY - for Von Neumann Entropy, you just need the eigenvalues (like
% for the norm) 

entropy = dysco_entropy(eigenvalues);



