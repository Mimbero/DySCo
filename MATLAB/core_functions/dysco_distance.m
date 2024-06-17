function distance = dysco_distance(matrix1,matrix2,what_distance)

% Giuseppe de Alteriis Oct 2023. 

% the distance is the norm of the difference of matrix1 and matrix2. 
% they are expressed as N eigenvectors, so every column of matrix1 or matrix2 
% is one of its eigenvectors. 

% NOTE NOTE! So matrix1 is not the matrix1 itself but its eigenvectors!
% Same for matrix2

% The norm of a matrix
% is based on the eigenvalues, so we just need the eigenvalues of the
% matrix difference=matrix1-matrix2. To do so we rely on the formula (1)
% see paper...

n_eigen = size(matrix1,2); % number of eigenvectors to express the matrix

% i need the eigenvalues of (matrix1-matrix2). I can use the formula (see
% paper)

minimatrix = zeros(2*n_eigen,2*n_eigen); %define minimatrix

% fill diagonal with the squared norms of the eigenvectors (these are
% actually the eigenvalues because in this representation the eigenvectors
% representing the matrix are weighed by the square root of their
% eigenvalues)

for i=1:n_eigen 
    minimatrix(i,i) = matrix1(:,i)'*matrix1(:,i);
    minimatrix(n_eigen+i,n_eigen+i) = -matrix2(:,i)'*matrix2(:,i); 
end

minimatrix_up_right = matrix1'*matrix2; % fill the rest with the scalar products

minimatrix(1:n_eigen,n_eigen+1:2*n_eigen) = minimatrix_up_right;
minimatrix(n_eigen+1:2*n_eigen,1:n_eigen) = -minimatrix_up_right';

if what_distance ~=2
    %now I can compute eigenvalues ... why...this is giving complex
    %numbers...??? Because the minimatrix in this case is not symmetric,
    %therefore matlab gives you complex numbers
    [~,lambdas] = eig(minimatrix);
    %lambdas = diag(lambdas);
    lambdas = real(diag(lambdas));
end

if what_distance==1
    distance = sum(abs(lambdas));
elseif what_distance==Inf
    distance = max(abs(lambdas));
elseif what_distance==2 % this should be equal to the sum of all squared elements of the matrix   
    distance = sqrt(sum(diag(minimatrix).^2)-2*sum(minimatrix_up_right.^2,'all'));

end

end