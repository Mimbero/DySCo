function [eigenvectors,eigenvalues] = compute_eigenvectors_sliding_corr(timeseries,half_window_size,n_eigen) 

% here I compute pearson correlation matrices using a sliding window. Then
% I take eigenvectors using the formula presented in the paper

% timeseries should be TxN, so every row is a timepoint and every column is
% a dimension

% here maybe we could add the case where we are full rank...?

if n_eigen > 2*half_window_size % maximum rank of corr matrix
   error('Number of requested eigenvectors is too large');
end

N = size(timeseries,2);
T = size(timeseries,1);

% matrix of eigenvectors stacked horizontally, one for each time point
eigenvectors = zeros(N,n_eigen,T-2*half_window_size); % so eigenvectors (:,:,t) are at time point t
% eigenvalues, one array of eigenvalues for each time point. I do
% T-2*half_window_size so that I exclude borders where the matrix is not
% well defined

eigenvalues = zeros(n_eigen,T-2*half_window_size);

for t=1:T-2*half_window_size

    truncated_timeseries = timeseries(t:t+2*half_window_size,:);
    zscored_truncated = zscore(truncated_timeseries); % as you can see from the formula (***) of the paper, 
                                                      % corr starts from
                                                      % summing the
                                                      % zscored
    normalizing_factor = size(truncated_timeseries,1)-1;                                              
    zscored_truncated = (1/sqrt(normalizing_factor))*zscored_truncated;
    
    % zscored_truncated contains all the vectors I am going to use for the
    % formula. So i have to scale them by 1/window_size-1. [maybe i SHALL
    % do it later at the end?]. 
    
    % ok, now that I have rearranged everything in a nice way I know that
    % the correlation matrix is zscored_truncated'*zscored_truncated.
    % However, instead of running an expensive EVD on it, now we can use
    % our formula
                                                     
    minimatrix = zscored_truncated*zscored_truncated';
    
     % Gathering eigenvectors (columns of v) and eigenvalues (diagonal of
     % a) of minimatrix. Eigenvalues will be them. Eigenvectors will be the
     % obtained coefficientsxoriginal vectors (see formula)
     
    [v,lambda] = eigs(minimatrix,n_eigen);
    
    % lambda are the eigenvalues I'm looking for:
    eigenvalues(:,t) = diag(lambda);
    
    % for the eigenvectors(see formula) I have to multiply the coefficients
    % with the original vectors
    eigenvectors(:,:,t) = zscored_truncated'*v;
   
    % normalising the eigenvectors by sqrt of eigenvalues so I can rebuild
    % matrix directly from them

    for i = 1:n_eigen
        eigenvectors(:,i,t) = eigenvectors(:,i,t)/norm(eigenvectors(:,i,t));
        eigenvectors(:,i,t) = eigenvectors(:,i,t) * sqrt(eigenvalues(i,t));
    end
end

end

