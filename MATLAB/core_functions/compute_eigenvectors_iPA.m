function [eigenvectors,eigenvalues] = compute_eigenvectors_iPA(timeseries)

% This function computes the timecourse of the 2 eigenvectors. They are scaled by sqrt of
% eigenvalues in order to recompose the full matrix. 

% timeseries must be a matrix. Each column is a signal. Each row is a time
% point. 

% detrend signal if needed 

n_channels = size(timeseries,2);
n = size(timeseries,1);

% compute hilbert transform for each signal Then compute phase locking matrix for each
% time point take eigenvector

for i=1:n_channels
    timeseries(:,i) = angle(hilbert(timeseries(:,i)));
end

eigenvectors = zeros(n_channels,2,n); % 2 because in iPA i always have 2 eigenvectors
eigenvalues = zeros(1,n);



for t=1:n

    % for each time point, compute the iPL matrix, and the c and s vectors (see paper) that 
    % allow the computation of eigenvalues and eigenvectors
    
    c = cos(timeseries(t,:))';
    s = sin(timeseries(t,:))';
    
    %I use analytical method to compute eigenvalues and eigenvectors (see
    % paper), by defining these quantitites
    
    sigma = s'*s;
    gamma = c'*c;
    xi = c'*s;
    delta = (gamma-sigma)^2+4*xi^2;
    B1 = ((sigma-gamma)+sqrt(delta))/(2*xi);
    B2 = ((sigma-gamma)-sqrt(delta))/(2*xi);
    
    v1 = c + B1*s;
    v2 = c + B2*s;
    v1 = v1/sqrt((v1'*v1)); % normalize eigenvectors. 
    v2 = v2/sqrt((v2'*v2));
    
    % find eigenvalues
    
    lambda1 = gamma+B1*xi;
    lambda2 = gamma+B2*xi;
    eigenvalues(t) = lambda1; % save biggest eigenvalue. The other is just n_channels-lambda1
    
    % scale eigenvectors by eigenvalues so that if I do v1*v1'+v2*v2' I
    % recreate the iPL matrix
    
    v1 = v1*sqrt(lambda1);
    v2 = v2*sqrt(lambda2);
    
    % we switch eigenvectors such that v1 is the leading one (the one with bigger eigenvalue) 

    if(lambda2>lambda1)
        tmp = v2;
        v2 = v1;
        v1 = tmp;
        eigenvalues(t) = lambda2;
    end
    
    % Eigenvectors multiplied by -1 are still eigenvectors. Thus, we invert
    % them if needed in order to have a timeseries of eigenvectors which
    % are all positively correlated. 

    if t>1 && (corr(v1,eigenvectors(:,1,t-1))<0)
        v1 = -v1;
    end
    
    if t>1 && (corr(v2,eigenvectors(:,2,t-1))<0)
        v2 = -v2;
    end
   
    % stack eigenvectors

    eigenvectors(:,1,t) = v1;
    eigenvectors(:,2,t) = v2;
    
end

% discard first and last timepoint as they are not meaningful 

eigenvectors = eigenvectors(:,:,2:end-1);
eigenvalues = eigenvalues(2:end-1);

end
