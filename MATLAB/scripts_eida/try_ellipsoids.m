%% TRY to see how iPL and ellipsoids relate!


%%

% try core eida scripts to compute sliding window eigenvalues and vectors of 
% any dataset 
clear
n_eigen = 2;
half_window_size = 20; % very small window 
%timeseries = readmatrix('prova_animale.txt');

%simulate timeseries
f = 0.1;
t = (0:0.1:10*pi)';
theta1 = 0.5*(1+cos(2*pi*f*t)); % slowly modulating phase
theta2 = -0.5*(1+cos(2*pi*f*t));
timeseries = [cos(2*pi*f*t) cos(2*pi*f*t+theta1) cos(2*pi*f*t+theta2)];


% select just 3 signals to visualize
timeseries = timeseries(:,1:3);

T = size(timeseries,1); %n time points
N = size(timeseries,2); % n signals


angles = angle(hilbert(timeseries));

%% try sliding window correlation

for t=1:T
    
    % try rebuild corr matrix with classical methods to see if method works
    
    lower = max(t-half_window_size,1); % bounds for window when it hits borders
    upper = min(t+half_window_size,T);
    truncated_timeseries = timeseries(lower:upper,:);
    
    theta1 = angles(t,1);
    theta2 = angles(t,2);
    theta3 = angles(t,3);
   
    taus = 0:0.01:2*pi;
    
    plot3(cos(taus-theta1),cos(taus-theta2),cos(taus-theta3),'color','red');
    hold on
    plot3(truncated_timeseries(:,1),truncated_timeseries(:,2),truncated_timeseries(:,3),'.','color','black');
    hold off
    pause(0.3);
    
end
