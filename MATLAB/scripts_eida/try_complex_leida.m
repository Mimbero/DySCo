% complex leida

% try core eida scripts to compute sliding window eigenvalues and vectors of 
% any dataset 
clear

Ts = 2.75;
lowest_freq = 0.01;
highest_freq = 0.08;

timeseries = readmatrix('prova_animale.txt');
timeseries = bandpass(timeseries,[lowest_freq highest_freq],1/Ts); % filter timeseries

T = size(timeseries,1); %n time points
N = size(timeseries,2); % n signals

theta = angle(hilbert(timeseries));

%% compute leading eigenv

leading_complex = exp(1i*theta)/sqrt(N);
leading_real = zeros(T,N);

% now compute leida and compare

for t=1:T
    
    ipl = cos(theta(t,:)'-theta(t,:));
    [l,~] = eigs(ipl,1);
    leading_real(t,:) = l';
    subplot(1,3,1)
    imagesc(ipl);
    title('IPL');
    subplot(1,3,2)
    imagesc(real(leading_complex(t,:)'*leading_complex(t,:)));
    title('complex LEiDA');
    subplot(1,3,3)
    imagesc(l*l');
    title('leading eigenv');
    pause(0.1);
end


%% now do clustering 

K=3;

[idx_real,C_real] = kmedoids(leading_real,K,'distance','cosine');
[idx_cx,C_cx] = kmedoids(leading_complex,K,'distance',@trace_complex);

% try visualize 

for k=1:K
    subplot(1,2,1)
    imagesc(C_real(k,:)'*C_real(k,:));
    title('real medoid');
    
    subplot(1,2,2)
    imagesc(real(C_cx(k,:)'*C_cx(k,:)));
    title('complex medoid')
    pause(2);
end


