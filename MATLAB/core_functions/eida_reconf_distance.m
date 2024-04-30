function distance = eida_reconf_distance(matrix1,matrix2,normalize)

% Giuseppe de Alteriis Oct 2023. 

% measure of reconfiguration (rotation of the eigenvectors in the multidim
% space)


n_eigen = size(matrix1,2); % number of eigenvectors to express the matrix

% for this formula (see paper) I need to normalise
if normalize
    for i=1:n_eigen
        matrix1(:,i)=matrix1(:,i)/norm(matrix1(:,i));
        matrix2(:,i)=matrix2(:,i)/norm(matrix2(:,i));
    end
end

minimatrix = matrix1'*matrix2; %define minimatrix
epsilon = minimatrix-eye(n_eigen);

distance = norm(minimatrix,'fro');

end

