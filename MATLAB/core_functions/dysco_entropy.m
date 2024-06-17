function von_neumann = dysco_entropy(eigenvalues) % calculates von neumann entropy starting from the 
% eigenvalue timeseries (each row is eigenvalue n. 1,2,3..., each column is
% a time point

% note that if the matrix has a lower rank than expected, and there's a row
% of null eigenvalues, they should be discarded 

    n_eigenvalues = size(eigenvalues,1);
    von_neumann = eigenvalues./repmat(sum(eigenvalues,1),n_eigenvalues,1);
    von_neumann = -sum(log(von_neumann).*von_neumann,1);
    
end