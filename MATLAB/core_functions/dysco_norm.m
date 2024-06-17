function norm = dysco_norm(eigenvalues,what_norm)

% Giuseppe de Alteriis Oct 2023. With the vector of eigenvalues of the
% matrix I can compute norm very simply

if what_norm==1
    norm = sum(abs(eigenvalues));
elseif what_norm==2 % this should be equal to the sum of all squared elements of the matrix
    norm = sqrt(sum(eigenvalues.^2));
elseif what_norm==Inf
    norm = max(eigenvalues);
end


end