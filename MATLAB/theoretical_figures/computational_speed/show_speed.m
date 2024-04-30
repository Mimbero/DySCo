%% define colors
clear
color_eida = '#00c79c';
color_numerical = '#394d5d';


%% run eigendecomposition with both our EiDA method and a classical method 
% that people would use 

T = 10;
n_trials = 50;
n_examples = 20;
sizes = round(logspace(2,4,n_trials));

mean_time_numerical = zeros(1,n_trials);
mean_time_anal = zeros(1,n_trials);
std_time_numerical = zeros(1,n_trials);
std_time_anal = zeros(1,n_trials);

for n=1:n_trials
    
    elapsed_times_numerical=zeros(1,n_examples);
    elapsed_times_anal=zeros(1,n_examples);
    for m=1:n_examples
        N = sizes(n);
        data = randn(N,T);
        
        % matlab method
        tic
        cov_matrix = cov(data');
        [eigen,lambda] = eigs(cov_matrix,T-1); % i ask all the eigenv, so t-1
        elapsed_times_numerical(m) = toc;
        
        % mont e pino 
        tic
        demeaned = data - mean(data,2);
        S = demeaned'*demeaned;
        [alphas,lambda] = eigs(S,T-1);
        eigen = demeaned*alphas;
        elapsed_times_anal(m) = toc;
    end
    
    mean_time_numerical(n) = mean(elapsed_times_numerical);
    mean_time_anal(n) = mean(elapsed_times_anal);
    std_time_numerical(n) = std(elapsed_times_numerical);
    std_time_anal(n) = std(elapsed_times_anal);
end

%%
figure
loglog(sizes,mean_time_anal,'color',color_eida,'linewidth',1.5);
hold on
loglog(sizes,mean_time_anal-0.5*std_time_anal,'color',color_eida,'linewidth',0.5);
hold on
loglog(sizes,mean_time_anal+0.5*std_time_anal,'color',color_eida,'linewidth',0.5);
hold on
loglog(sizes,mean_time_numerical,'color',color_numerical,'linewidth',1.5);
hold on
loglog(sizes,mean_time_numerical-0.5*std_time_numerical,'color',color_numerical,'linewidth',0.5);
hold on
loglog(sizes,mean_time_numerical+0.5*std_time_numerical,'color',color_numerical,'linewidth',0.5);

grid on

%legend('EiDA','','','Numerical','','');