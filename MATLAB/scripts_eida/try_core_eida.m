% try core eida scripts to compute sliding window eigenvalues and vectors of 
% any dataset 

n_eigen = 2;
half_window_size = 10; % so the window is symmetric and long 21 elements

% timeseries = randn(200,100); % simulated
timeseries = readmatrix('prova_animale.txt');
T = size(timeseries,1); %n time points
N = size(timeseries,2); % n signals


%% try sliding window correlation

addpath('./core_functions');
[v,l]=compute_eigenvectors_sliding_corr(timeseries,half_window_size,n_eigen);

for t=1:T
    
    % try rebuild corr matrix with classical methods to see if method works
    
    lower = max(t-half_window_size,1); % bounds for window when it hits borders
    upper = min(t+half_window_size,T);
    truncated_timeseries = timeseries(lower:upper,:);
    literature_corr = corrcoef(truncated_timeseries);
    [v_lit,l_lit] = eigs(literature_corr,n_eigen);
    
    rebuilt_matrix = v(:,:,t)*v(:,:,t)';
    
    % i scale them to be unitary so that I can see how they correlate with
    % the "Numerical" eigenvectors
    
    for i=1:n_eigen
        v(:,i,t) = v(:,i,t)/sqrt(l(i,t));
    end
    
    subplot(2,2,1)
    imagesc(literature_corr);
    title('empirical')
    subplot(2,2,2)
    imagesc(rebuilt_matrix);
    title('rebuilt with our formula')
    subplot(2,2,3)
    scatter(l(:,t),diag(l_lit));
    title('our eigenval versus empirical')
    subplot(2,2,4)
    imagesc(v(:,:,t)'*v_lit);
    title('our eigenvect versus empirical')
    
    pause(0.1);
    
end





