%% try eida norm - looks like it works!!!

n_trials = 1000;

norm_literature=zeros(n_trials,1);
norm_my_code=zeros(n_trials,1);

for t=1:n_trials
    
    
    N = 50;
    matrix = zeros(N,N);
    
    % build square simm matrix
    for i=1:N
        vector1 = randn(N,1);
        matrix = matrix+vector1*vector1';
    end
    
    % ok now we have the matrix. now we need eigenvalues
    
    [v,d] = eig(matrix);
    d = diag(d);
    
    norm_literature(t)=sum(matrix.^2,'all');
    norm_my_code(t) = eida_norm(d,2);
    
end

scatter(norm_my_code,norm_literature);

%% now more complicated - try eida distance

clear
addpath('./core_functions');
n_trials = 1000;
N = 50;

dist_literature=zeros(n_trials,1);
dist_my_code=zeros(n_trials,1);
dist_my_code_approx=zeros(n_trials,1);


for t=1:n_trials
    
    matrix1 = zeros(N,N);
    matrix2 = zeros(N,N);
    
    % build square simm matrix
    for i=1:N
        % if u want add weight to make it more naturalistic
        %weight = 1;
        weight = 0.75^i;
        vector1 = randn(N,1);
        matrix1 = matrix1+weight*(vector1*vector1');
        vector2 = randn(N,1);
        matrix2 = matrix2+weight*(vector2*vector2');
    end
    
    % ok now we have the matrix.
    dist_literature(t) = norm(matrix1(:)-matrix2(:),2);
    
    % to compute eida distance I need eigenvector representation. I need
    % eigenvectors to be scaled by sqrt of their eigenvalues
    
    [v1,lambda1] = eig(matrix1);
    [lambda1, idx1] = sort(diag(lambda1), 'descend'); %sort them
    v1 = v1(:, idx1);
    
    [v2,lambda2] = eig(matrix2);
    [lambda2, idx2] = sort(diag(lambda2), 'descend');
    v2 = v2(:, idx2);
    
    % scale them as required
    for i=1:N
        v1(:,i) = (v1(:,i)/norm(v1(:,i)))*sqrt(lambda1(i));
        v2(:,i) = (v2(:,i)/norm(v2(:,i)))*sqrt(lambda2(i));
    end
    
    dist_my_code(t) = (eida_distance(v1,v2,2));
    
    % good approximation with just 2 eigens (instead of
    % 100)...note...sometimes eigenvalues are increasing,sometimes
    % decreasing...why???
    
    dist_my_code_approx(t) = (eida_distance(v1(:,1:4),v2(:,1:4),2));
    
end


scatter(dist_my_code,dist_literature);
figure
scatter(dist_my_code_approx,dist_literature);
