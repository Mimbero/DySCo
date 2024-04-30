% this script generates a timeseries with known underlying dynamic
% connectivity pattern. To do so, given a connectivity pattern A, it is
% sufficient to multiply a independent identically distributed timeseries
% and multiply it by L, where L is the cholesky decomposition of A, which
% is always feasible because A is symmetric. We do so and then we
% concatenate signals with different connectivity patterns in order to see
% if we can then cluster and correctly retrieve them

clear, close

N = 10; % number of signals 

%%

% let us suppose we have k dynamic connectivity clusters. We will generate
% a dataset with signals and k underlying covariance patterns

k = 5; % number of clusters
T = 100; % number of time steps for each chunk of signals with that underly
         % ing covariance pattern
 
C = cell(1,k);
timeseries = zeros(T*k,N);
         
for i = 1:k
    z = randn(N,T); % indep identically distributed data 
    C{i} = generate_observable(N); % generate a symmetric positive semidef
                                % matrix as possible covariance matrix
    L = chol(C{i}); 
    signals = L'*z; % cholesky*iid signals gives us signals with covariance
                    % matrix = C
    timeseries((i-1)*T+1:i*T,:) = signals';
end
             
% check if the obtained timeseries has the required dFC pattern

figure
for i=1:k
    subplot(k+1,2,2*i-1)
    imagesc(corrcoef(timeseries((i-1)*T+1:i*T,:)));
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
title('generated timeseries');
axis off

%%

% now that we have this synthetic dataset with underlying patterns, let us
% run the clustering algorithms in order to retrieve them. We will try
% classic kmeans on full matrix distance vs leida vs trace distance  

half_wind_size = 5;

dFC = zeros(k*T,0.5*N*(N+1));
% codice ancora pezzotto da commentare...save in uppertri...ritrovare
% funzioni uppertri - stasera vedere 

for t=1:k*T
    lower = max(1,t-half_wind_size);
    upper = min(k*T,t+half_wind_size);
    
    chunk = timeseries(lower:upper,:);
    dFC(t,:) = uppertri(corrcoef(chunk))';
  
end


%%
% try medoids different distances - thm; our new algo works; leida fails
idx_medoids = kmedoids(dFC,k,'Distance','cosine');

idx_medoids_trace = kmedoids(dFC,k,'Distance',@trace_distance);
idx_medoids_norm2 = kmedoids(dFC,k,'Distance',@norm2_distance);
idx_medoids_leida = kmedoids(dFC,k,'Distance',@eig_distance);
% then we will try the same with dbscan

% thm i'm sorry but looks like the only way is normal distance. However, we
% can still use trace as a good approximation


plot(idx_medoids);
hold on
%plot(idx_medoids_trace);
hold on
plot(idx_medoids_norm2);
hold on
%plot(idx_medoids_leida);
legend('cosine','trace','norm2','leida');

%%
% try same with kmeans
[idx,~,~] = kmeans_custom_distance(dFC,k,200,50,[],true,@cosine_distance);

[idx_trace,~,~] = kmeans_custom_distance(dFC,k,200,50,[],true,@trace_distance);
[idx_norm2,~,~] = kmeans_custom_distance(dFC,k,200,50,[],true,@norm2_distance);
[idx_leida,~,~] = kmeans_custom_distance(dFC,k,200,50,[],true,@eig_distance);
% then we will try the same with dbscan

% thm i'm sorry but looks like the only way is normal distance. However, we
% can still use trace as a good approximation


plot(idx);
hold on
%plot(idx_trace);
hold on
plot(idx_norm2);
hold on
%plot(idx_leida);
legend('cosine','trace','norm2','leida');

%% see fcd
for i=1:50
for j=1:50
matrix_trace(i,j)=trace_distance(dFC(i*10,:),dFC(j*10,:));
matrix_corr(i,j)=1-abs(corr(dFC(i*10,:)',dFC(j*10,:)'));
matrix_fidelity(i,j)=fidelity(dFC(i*10,:),dFC(j*10,:));
end
end
figure
subplot(1,3,1);
imagesc(matrix_trace)
subplot(1,3,2);
imagesc(matrix_corr)
subplot(1,3,3);
imagesc(matrix_fidelity)

%% try if reconf speed with the 2 are similar

for i=1:50
reconf_trace(i)=trace_distance(dFC(i,:),dFC(i+1,:));
reconf_corr(i)=1-abs(corr(dFC(i,:)',dFC(i+1,:)'));
end

plot(reconf_trace);
hold on
plot(reconf_corr);

corr(reconf_trace',reconf_corr')