% this script generates a timeseries with known underlying dynamic
% connectivity pattern. To do so, given a connectivity pattern A, it is
% sufficient to multiply a independent identically distributed timeseries
% and multiply it by L, where L is the cholesky decomposition of A, which
% is always feasible because A is symmetric. We do so and then we
% concatenate signals with different connectivity patterns 

% then, we apply the EiDA functions and see if we can recover this

addpath('../../core_functions');
addpath('../../scripts_eida');

clear, close

N = 10; % number of signals 

%%

% let us suppose we have k dynamic connectivity clusters. We will generate
% a dataset with signals and k underlying covariance patterns

k = 5; % number of clusters
T = 1000; % number of time steps for each chunk of signals with that underly
         % ing covariance pattern
 
C = cell(1,k);
timeseries = zeros(T*k,N);
         
for i = 1:k
    z = randn(N,T); % indep identically distributed data
    % OPTIONAL: lowPASS
    %z = lowpass(z',0.05,1)';
    C{i} = generate_observable(N); % generate a symmetric positive semidef
                                     % matrix as possible covariance matrix
                                     % multiply it by i so we have
                                     % increasing norms of the matrices
    L = chol(C{i}); 
    signals = L'*z; % cholesky*iid signals gives us signals with covariance
                    % matrix = C
    timeseries((i-1)*T+1:i*T,:) = signals';
end
             
% check if the obtained timeseries has the required dFC pattern

figure
for i=1:k
    subplot(k+1,2,2*i-1)
    imagesc(cov(timeseries((i-1)*T+1:i*T,:)));
    if i==1 title('estimated covariance'); end
    axis off
    subplot(k+1,2,2*i)
    imagesc(C{i});
    if i==1 title('true covariance'); end
    axis off
end

%last row, show signals
i = i+1;
subplot(k+1,2,[2*i-1 2*i])
plot(timeseries);
hold on
xline(T*(1:k-1),'color','red','linewidth',2);
title('generated timeseries');
axis off

%% now compute sliding window covariance

half_window = 60;
n_eigen = N-1;

[eigenvectors,eigenvalues] = compute_eigenvectors_sliding_cov(timeseries,half_window,n_eigen);

% show some estimated sliding window covariance matrix versus real cov
% matrix

figure
for i=1:k
    subplot(k,5,5*(i-1)+1)
    imagesc(C{i});
    if i==1 title('true covariance'); end
    axis off
    for j=1:4
        subplot(k,5,5*(i-1)+j+1)
        index = (i-1)*T+120*j-half_window; % extract index for a specific frame
        imagesc(eigenvectors(:,:,index)*eigenvectors(:,:,index)');
        axis off
    end
    
end

%% now compute time-varying norm to see if it captures the change

norm1 = [zeros(1,half_window) eida_norm(eigenvalues,1) zeros(1,half_window)];
norm2 = [zeros(1,half_window) eida_norm(eigenvalues,2) zeros(1,half_window)];
norminf = [zeros(1,half_window) eida_norm(eigenvalues,Inf) zeros(1,half_window)];


figure
plot(norm1);
hold on
plot(norm2);
hold on
plot(norminf);
hold on
xline(T*(1:k),'color','red','linewidth',2);
legend('1','2','inf','separation');

%% now compute time-varying distance = reconfiguration speed on normalized matrices

delay = 100;

speed2 = zeros(1,T*k);
total_time = T*k-2*half_window;

for t=1:total_time-delay
    %not normalize
    normt = 1;
    normtd = 1;
    %normalize
    normt = sum(eigenvalues(:,t).^2);
    normtd = sum(eigenvalues(:,t+delay).^2);
    speed2(half_window+t)=eida_distance(eigenvectors(:,:,t)/sqrt(normt),eigenvectors(:,:,t+delay)/sqrt(normtd),2);
end

figure
%plot(smooth(speed2,5));
plot(speed2);
xline(T*(1:k-1),'--','color','red','linewidth',0.5);
legend('normalized distance','separation');

%% try new distance - multiaxis rotation

delay = 40;

speed3 = zeros(1,T*k);
total_time = T*k-2*half_window;

for t=1:total_time-delay
    speed3(half_window+t)=eida_reconf_distance(eigenvectors(:,:,t),eigenvectors(:,:,t+delay));
end

figure
%plot(smooth(speed2,5));
plot(speed3,'color','green','linewidth',2);
xline(T*(1:k-1),'--','color','red','linewidth',0.5);
legend('reconfiguration distance','separation');

%% try plotting fcd matrix to see what happens...

subsampling = 10;

fcd1 = zeros(total_time/subsampling,total_time/subsampling);
fcd2 = zeros(total_time/subsampling,total_time/subsampling);
fcdinf = zeros(total_time/subsampling,total_time/subsampling);


for i=1:total_time/subsampling
    for j=1:total_time/subsampling
        fcd1(i,j) = eida_distance(eigenvectors(:,:,i*subsampling),eigenvectors(:,:,j*subsampling),1);
        fcd2(i,j) = eida_distance(eigenvectors(:,:,i*subsampling),eigenvectors(:,:,j*subsampling),2);
        fcdinf(i,j) = eida_distance(eigenvectors(:,:,i*subsampling),eigenvectors(:,:,j*subsampling),Inf);
    end
end

%% finally try clustering to see if you can recover the cov patterns

% to do clustering I have to reshape my eigenvalues accordingly

eig_for_clustering = reshape(eigenvectors,N*n_eigen,total_time)';

[idx,Centroids] = kmedoids(eig_for_clustering,k,'Distance',@eida_dist4clustering,'Options',statset('Useparallel',true));
figure
plot([zeros(half_window,1);idx; zeros(half_window,1)]);

%% note: if you want to recover workspace for exact figures, run it



%% do overall figure for the paper!

%show signals
figure
subplot(4,1,1)
plot(timeseries);
hold on
xline(T*(1:k-1),'color','red','linewidth',2);
axis off

%show reconf speed
subplot(4,1,3)
plot(speed2,'linewidth',1,'color','blue');
axis off

%show indexes for clustering
subplot(4,1,4)
plot(idx,'linewidth',1,'color','black');
axis off

%% show the k true covariances

figure 
for i=1:k
    subplot(1,k,i)
    imagesc(C{i});
    axis off
end

%% show a "line" of dynamic matrices to show the evolution

figure
h=0;
for i=1:k
    for j=1:4
        h=h+1;
        subplot(1,k*4,h);
        index = (i-1)*T+120*j-half_window; % extract index for a specific frame
        if j==4 && i~=k % I want to show the matrices in the middle
            index = i*T-half_window;
        end
        imagesc(eigenvectors(:,:,index)*eigenvectors(:,:,index)');
        axis off
    end
    
end

%% show the k medoids for visualization

figure 
for i=1:k
    subplot(1,k,i)
    M = reshape(Centroids(i,:),N,n_eigen);
    imagesc(M*M');  
    axis off
end


%% now show fcd

% decide colormap
colormap spring

figure
subplot(1,3,1)
imagesc(real(fcd1)); % WHY THIS IS COMPLEX????
colorbar
axis off
subplot(1,3,2)
imagesc(real(fcd2)); % WHY THIS IS COMPLEX????
colorbar
axis off
subplot(1,3,3)
imagesc(real(fcdinf)); % WHY THIS IS COMPLEX????
colorbar
axis off

%% save workspace for replicability if required

save('workspace_fig_17_nov');
