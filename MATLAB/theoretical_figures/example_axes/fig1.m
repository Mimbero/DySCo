% generate 2 timeseries with 2 different covariance patterns and show that
% if you concatenate them and see a unique pattern it won't work anymore 
clear 

N = 3;
T = 1000;


timeseries1= randn(N,T);
timeseries2= randn(N,T);



% toy example covariance patterns
cov1 = 10*[1;1;0]*[1 1 0] + [0;0;1]*[0 0 1] + 0.1*[1;-1;0]*[1 -1 0];
cov2 = 10*[1;0;1]*[1 0 1] + [0;1;0]*[0 1 0] + 0.1*[-1;0;1]*[-1 0 1];

%imagesc(cov2);

% then instead of chol with eps you can use evd
timeseries1 = chol(cov1)'*timeseries1;
timeseries2 = chol(cov2)'*timeseries2;

%%

scatter3(timeseries1(1,:),timeseries1(2,:),timeseries1(3,:), 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'blue', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeAlpha', 0.1);
hold on
scatter3(timeseries2(1,:),timeseries2(2,:),timeseries2(3,:), 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'red', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeAlpha', 0.1);

[v1,l1] = eigs(cov1);
[v2,l2] = eigs(cov2);

v1 = v1*l1;
v2 = v2*l2;

quiver3([0 0 0],[0 0 0],[0 0 0],v1(1,:),v1(2,:),v1(3,:),'color','blue','linewidth',2);
hold on
quiver3([0 0 0],[0 0 0],[0 0 0],v2(1,:),v2(2,:),v2(3,:),'color','red','linewidth',2);
hold on

% now plot the overall corr pattern

cov_tot = cov([timeseries1';timeseries2']);
[v_tot,l_tot] = eigs(cov_tot);
v_tot = v_tot*l_tot;
quiver3([0 0 0],[0 0 0],[0 0 0],v_tot(1,:),v_tot(2,:),v_tot(3,:),'color','black','linewidth',2);
hold on

%plot axes as a ref
quiver3([-10 -10 -10],[-10 -10 -10],[-10 -10 -10],[5 0 0],[0 5 0],[0 0 5],'color','black','linewidth',2);
axis off
% imagesc(cov([timeseries1';timeseries2']));

%%

colormap jet
subplot(1,3,1)
imagesc(corrcoef(timeseries1'),[0,1]);
colorbar
subplot(1,3,2)
imagesc(corrcoef(timeseries2'),[0,1]);
colorbar
subplot(1,3,3)
imagesc(corrcoef([timeseries1';timeseries2']),[0,1]);
colorbar

%%

plot(timeseries1');
figure
plot(timeseries2');
