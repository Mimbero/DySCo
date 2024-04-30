% generate 2 timeseries with 2 different covariance patterns and show that
% if you concatenate them and see a unique pattern it won't work anymore 
clear 

%% set colors

color_low_entropy = '#1836b2';
color_high_entropy = '#36a9e1';

N = 3;
T = 1000;


timeseries1= randn(N,T);
timeseries2= randn(N,T);



% toy example covariance patterns. 1 is a "rich" timeseries, 2 is a "non
% rich" timeseries
cov1 = eye(3);
cov2 = cov1;
cov2(1,1) = 10;
cov2(3,3) = 0.5;

%imagesc(cov2);

% then instead of chol with eps you can use evd
timeseries1 = chol(cov1)'*timeseries1;
timeseries2 = chol(cov2)'*timeseries2;

%%

[v1,l1] = eigs(cov1);
[v2,l2] = eigs(cov2);

v1 = v1*l1;
v2 = v2*l2;

figure
scatter3(timeseries1(1,:),timeseries1(2,:),timeseries1(3,:), 'MarkerFaceColor', color_low_entropy, 'MarkerEdgeColor', '#808588', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeAlpha', 0.1);
hold on
quiver3([0 0 0],[0 0 0],[0 0 0],v1(1,:),v1(2,:),v1(3,:),'color',color_low_entropy,'linewidth',2);
axis equal
axis off
figure
scatter3(timeseries2(1,:),timeseries2(2,:),timeseries2(3,:), 'MarkerFaceColor', color_high_entropy, 'MarkerEdgeColor', '#808588', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeAlpha', 0.1);
hold on
quiver3([0 0 0],[0 0 0],[0 0 0],v2(1,:),v2(2,:),v2(3,:),'color',color_high_entropy,'linewidth',2);
axis equal
axis off

%% show barplots of eigenvalues 

figure 
subplot(1,2,1)
bar(diag(l1),'facecolor','white','edgecolor',color_low_entropy,'linewidth',3);
axis off
subplot(1,2,2)
bar(diag(l2),'facecolor','white','edgecolor',color_high_entropy,'linewidth',3);
axis off


