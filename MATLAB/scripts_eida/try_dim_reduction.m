% try the new dimensionality reduction method versus the classic one 
% nelly furtado say it right

% altra cosa: strano che con correlation funziona peggio...probly c'è un
% errore...

clear,close

% choose some input data 
input_data = readmatrix('prova_animale.txt')';
input_data = zscore(input_data,0,2); % if you zscore...also the correlation estimation works better...but cov is probly worse? check...
% input_data = randn(200,500);
% C = generate_observable(200);
% L = chol(C); 
% input_data = L'*input_data;

%% 
N = size(input_data,1);
T = size(input_data,2);
tot_n_comp = 44;
half_wind_size = 7; 

% Overall step: Mean center the data. Make sure you are demeaning in the
% right axis! - to check
mean_data = mean(input_data,2);
timeseries = input_data - mean_data;

% we will work only with "centered data timeseries"

%% first of all we build the ground truth dFC correlation and covariance matrices

dFCov_truth = zeros(N,N,T);

for t=1:T
    lower_bound = max(1,t-half_wind_size);
    upper_bound = min(T,t+half_wind_size);
    
    chunk = timeseries(:,lower_bound:upper_bound)';
    dFCov_truth(:,:,t) = cov(chunk); % we work with covariance, we'll rebuild correlation later
end

dFCorr_truth = get_corr_from_cov(dFCov_truth);


%% OUR METHOD: do one single "overall" PCA IN THE NATIVE SPACE OF THE SIGNALS, 
% then see how you change performance in
% reconstructing dFC by changing n of components

% Step 2: Calculate the covariance matrix
covariance_matrix = cov(timeseries');

% Step 3: Calculate the eigenvectors and eigenvalues for PCA
[eigenvectors, eigenvalues] = eig(covariance_matrix);

% Sort eigenvectors by decreasing eigenvalues
[eigenvalues_sorted, index] = sort(diag(eigenvalues), 'descend');
eigenvectors_sorted = eigenvectors(:, index);

% ok now I have the linear transformation for PCA, let us change the n of
% components and see how the performance in reconstructing dFC changes

mse_corr = zeros(1,tot_n_comp);
mse_cov = zeros(1,tot_n_comp);
r2_corr = zeros(1,tot_n_comp);
r2_cov = zeros(1,tot_n_comp);

for num_components = 1:tot_n_comp
    
    % Select the top 'num_components' eigenvectors. This is the A
    % transformation from signals to "reduced" signals
    A = eigenvectors_sorted(:, 1:num_components)';
    
    % Project data onto the new reduced space
    reduced_data = A*timeseries;
    
    % Now compute the dFC "mini" matrices using a sliding window
    
    dFCov_mini = zeros(num_components,num_components,T);

    for t=1:T
        lower_bound = max(1,t-half_wind_size);
        upper_bound = min(T,t+half_wind_size);
        
        chunk = reduced_data(:,lower_bound:upper_bound)';
        dFCov_mini(:,:,t) = cov(chunk); % we work with covariance, we'll rebuild correlation later
    end
    
    % ok now we have the dFC in the reduced space. We want to rebuild dFC
    % in the "actual" space using our formula and see how the approximation performs
    
    dFCov_approx = zeros(size(dFCov_truth));
    
    pinvA = pinv(A);
    pinvAt = pinv(A');
    
    for t=1:T
        %dFCov_approx(:,:,t) = pinvA*dFCov_mini(:,:,t)*pinvAt; %this is the
        %original code. I'm trying now 2 options: transpose and pinv
        
        dFCov_approx(:,:,t) = A'*dFCov_mini(:,:,t)*A; %method with transpose
    end
    
    dFCorr_approx = get_corr_from_cov(dFCov_approx);
    
    % compute mean squared error
    mse_cov(num_components) = mean(((dFCov_truth-dFCov_approx).^2),'all');
    mse_corr(num_components) = mean(((dFCorr_truth-dFCorr_approx).^2),'all');
    
    % compute mean rsquared
    r_cov = zeros(1,T);
    r_corr = zeros(1,T);
    
    for t=1:T
        vectorized_cov_truth = dFCov_truth(:,:,t);
        vectorized_corr_truth = dFCorr_truth(:,:,t);
        vectorized_cov_approx = dFCov_approx(:,:,t);
        vectorized_corr_approx = dFCorr_approx(:,:,t);
        
        r_cov(t) = corr(vectorized_cov_approx(:),vectorized_cov_truth(:));
        r_corr(t) = corr(vectorized_corr_approx(:),vectorized_corr_truth(:));
    end
    
    r2_cov(num_components) = mean(r_cov.^2);
    r2_corr(num_components) = mean(r_corr.^2);
    
end


%% benchmark method - here I vectorize the matrix and I do a PCA on it - let's start with cov

% reshape the matrix to vectorise it, so I can apply classic PCA
vectorized_cov_truth = zeros(N*(N+1)/2,T);

for t=1:T
    vectorized_cov_truth(:,t) = uppertri(dFCov_truth(:,:,t));
end

% demean it to do PCA
mean_to_add = mean(vectorized_cov_truth,2);
vectorized_cov_demeaned = vectorized_cov_truth-mean_to_add;

% perform PCA 
covariance_matrix = cov(vectorized_cov_demeaned');
% Calculate the eigenvectors and eigenvalues for PCA
[eigenvectors, eigenvalues] = eig(covariance_matrix);

% Sort eigenvectors by decreasing eigenvalues
[eigenvalues_sorted, index] = sort(diag(eigenvalues), 'descend');
eigenvectors_sorted = eigenvectors(:, index);

% ok now I have the linear transformation for PCA, let us change the n of
% components and see how the performance in reconstructing dFC changes

mse_cov_benchmark = zeros(1,tot_n_comp);
r2_cov_benchmark = zeros(1,tot_n_comp);

for num_components = 1:tot_n_comp
    
    % Select the top 'num_components' eigenvectors. This is the A
    % transformation from signals to "reduced" signals
    A = eigenvectors_sorted(:, 1:num_components)';
    
    % Project data onto the new reduced space
    reduced_cov = A*vectorized_cov_demeaned;
    reconstructed_cov = A'*reduced_cov+mean_to_add;
    
    mse_cov_benchmark(num_components) = mean(((vectorized_cov_truth-reconstructed_cov).^2),'all');
    
    r_cov = zeros(1,T); % mean rsquared
    for t=1:T
        r_cov(t) = corr(vectorized_cov_truth(:,t),reconstructed_cov(:,t));
    end
    
    r2_cov_benchmark(num_components) = mean(r_cov.^2);
    
end


%% do the same with corr

% reshape the matrix to vectorise it, so I can apply classic PCA
vectorized_corr_truth = zeros(N*(N+1)/2,T);

for t=1:T
    vectorized_corr_truth(:,t) = uppertri(dFCorr_truth(:,:,t));
end

% demean it to do PCA
mean_to_add = mean(vectorized_corr_truth,2);
vectorized_corr_demeaned = vectorized_corr_truth-mean_to_add;

% perform PCA 
covariance_matrix = cov(vectorized_corr_demeaned');
% Calculate the eigenvectors and eigenvalues for PCA
[eigenvectors, eigenvalues] = eig(covariance_matrix);

% Sort eigenvectors by decreasing eigenvalues
[eigenvalues_sorted, index] = sort(diag(eigenvalues), 'descend');
eigenvectors_sorted = eigenvectors(:, index);

% ok now I have the linear transformation for PCA, let us change the n of
% components and see how the performance in reconstructing dFC changes

mse_corr_benchmark = zeros(1,tot_n_comp);
r2_corr_benchmark = zeros(1,tot_n_comp);


for num_components = 1:tot_n_comp
    
    % Select the top 'num_components' eigenvectors. This is the A
    % transformation from signals to "reduced" signals
    A = eigenvectors_sorted(:, 1:num_components)';
    
    % Project data onto the new reduced space
    reduced_corr = A*vectorized_corr_demeaned;
    reconstructed_corr = A'*reduced_corr+mean_to_add;
    mse_corr_benchmark(num_components) = mean(((vectorized_corr_truth-reconstructed_corr).^2),'all');
    
    r_corr = zeros(1,T); % mean rsquared
    for t=1:T
        r_corr(t) = corr(vectorized_corr_truth(:,t),reconstructed_corr(:,t));
    end
    
    r2_corr_benchmark(num_components) = mean(r_corr.^2);
    
end

%% 
figure
subplot(2,2,1)
plot(mse_cov_benchmark)
hold on
plot(mse_cov)
legend('benchmark','pinuozzo');
title('covariance matrix');
ylabel('mse')

subplot(2,2,2)
plot(mse_corr_benchmark)
hold on
plot(mse_corr)
legend('benchmark','pinuozzo');
title('correlation matrix');


subplot(2,2,3)
plot(r2_cov_benchmark)
hold on
plot(r2_cov)
legend('benchmark','pinuozzo');
ylabel('r2')

subplot(2,2,4)
plot(r2_corr_benchmark)
hold on
plot(r2_corr)
legend('benchmark','pinuozzo');



% find application/example to convince
% ex: data driven parcellation
% for example rest-task-rest-task

% causal perturbations in the reduced space
