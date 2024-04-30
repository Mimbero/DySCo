function dFCorr = get_corr_from_cov(dFCov)

% transform dFC covariance matrices in correlation matrices

N = size(dFCov,1);
T = size(dFCov,3);

dFCorr = dFCov;

for t=1:T
    
    % at each time point extract sqrt of diagonal
    diag = zeros(N,1);
    for i=1:N
    diag(i) = sqrt(dFCov(i,i,t));
    end

    dFCorr(:,:,t) = dFCorr(:,:,t)./(diag*diag');
    
end

end