addpath('../core_functions/');
addpath('../../../data/');

eigenvectors = load('eigenvectors.mat').eigenvectors;
eigenvalues = load('eigenvalues.mat').eigenvalues;
task = readmatrix('TaskTC.txt'); 

T = size(eigenvectors,1);

%%
fcd = zeros(T,T);

for i=1:T
    for j=i:T
        fcd(i,j) = real(eida_distance(squeeze(eigenvectors(i,:,:)),squeeze(eigenvectors(j,:,:)),2));
    end
end


%%

task_reduced = mean(task,2);
task_reduced = task_reduced(16:390);

derivative = task_reduced(2:end)-task_reduced(1:end-1);


figure
subplot(4,3,1:9)
imagesc(fcd_reconf)
ax1 = gca;
subplot(4,3,10:12)
%plot(task_reduced);
hold on
plot([0; derivative]);
ax2 = gca;
linkaxes([ax1 ax2],'x');

figure 
subplot(2,1,1)
mean_dist = mean(fcd_reconf);
plot(mean_dist(2:end));
subplot(2,1,2)
plot(abs(derivative));
title(corr(derivative(10:end-9),mean_dist(11:end-9)'));

%%

task_reduced = mean(task,2);
task_reduced = task_reduced(16:390);

von_neumann = eigenvalues./repmat(sum(eigenvalues,1),5,1);
von_neumann = -sum(log(von_neumann).*von_neumann,1);

figure 
subplot(2,1,1)
plot(task_reduced);
ax1 = gca;
subplot(2,1,2)
plot(von_neumann);
ax2 = gca;
title(corr(von_neumann',task_reduced));

linkaxes([ax1 ax2],'x');
%%
% try different correlations

correlations = zeros(1,18);
for i=1:18
    correlations(i) = corr(von_neumann(20:end-20)',task_reduced(20-i:end-20-i));
end

figure
plot(correlations)
title('shift that maximises correlation');

% take home message: there's an intrinsic lag that creates a temporal
% shift...we should discuss this...

%% try reconfiguration speed - looks like it works with eigen reconf...

lag = 10;

reconf_speed = zeros(1,375);

for i = lag+1:375
    reconf_speed(i)=real(eida_reconf_distance(squeeze(eigenvectors(i,:,:)),squeeze(eigenvectors(i-lag,:,:)),1));
end

figure
subplot(2,1,1)
plot(smooth(reconf_speed,5));
ax1 = gca;
subplot(2,1,2)
plot(task_reduced);
ax2 = gca;
title(corr(reconf_speed(lag:end)',task_reduced(lag:end)));
linkaxes([ax1 ax2],'x');


