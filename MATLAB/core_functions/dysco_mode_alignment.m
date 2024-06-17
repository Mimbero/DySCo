function distance = dysco_mode_alignment(matrix1,matrix2)

% Giuseppe de Alteriis Oct 2023.

% measure of reconfiguration (rotation of the eigenvectors in the multidim
% space)


n_eigen = size(matrix1,2); % number of eigenvectors to express the matrix

% for this formula (see paper) I need to normalise

for i=1:n_eigen
    matrix1(:,i)=matrix1(:,i)/norm(matrix1(:,i));
    matrix2(:,i)=matrix2(:,i)/norm(matrix2(:,i));
end


minimatrix = matrix1'*matrix2; %define minimatrix
frob_product = norm(minimatrix,'fro');

distance = 2*(n_eigen-frob_product);

end

