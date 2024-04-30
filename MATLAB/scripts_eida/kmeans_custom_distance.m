% this code does definitely still have problems, but it is enough to try to
% see if it works...

function [idx_out,C_out,Obj_out] = kmeans_custom_distance(X,K,maxIter,nRepl,seed,parallel,distancefun)

% [idx,C,Obj] = pdfc_diametrical_clustering(X,K,maxIter,nRepl,init,seed)
% Diametrical clustering as in Sra2012 algorithm 2.
% Input:
%%% X - nxp data
%%% K - number of clusters K
% Optional input:
%%% maxIter (default 1000)
%%% nRepl (default 5)
%%% init (default '++', other option 'uniform')
% Output:
%%% idx - state sequence (closest centroid for all samples)
%%% C - computed centroids
%
% Sra2012: "The multivariate Watson distribution: Maximum-likelihood 
% estimation and other aspects", Sra S, Karp D, 2012.
%
% Anders S Olsen June - November 2021, Neurobiology Research Unit
%
% Giuseppe de Alteriis, July 2022, EiDA


[n,p] = size(X);

if nargin == 2
    maxIter = 1000;
    nRepl = 5;
    init = 'plus';
    seed = [];
    parallel = false;
elseif nargin == 3
    nRepl = 5;
    init = 'plus';
    seed = [];
    parallel = false;
elseif nargin == 4
    init = 'plus';
    seed = [];
    parallel = false;
elseif nargin == 5
    seed = [];
    parallel = false;
elseif nargin == 6
    parallel = false;
end

if ~isempty(seed)
    s = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(s);
    stream = RandStream.getGlobalStream();
else
    stream = [];
end

if size(X,1)<size(X,2)
    error('Wrong input data format, should be NxP')
end

if parallel
    numWorkers = min([12,nRepl]);
else
    numWorkers = 0;
end


% Initialize variables 

C_final = zeros(p,K,nRepl);
objective_final = zeros(1,nRepl);
X_part_final = zeros(n,nRepl);

% perform clustering
%for repl = 1:nRepl
parfor (repl = 1:nRepl,numWorkers) 
    objective = zeros(maxIter,1);
    partsum = zeros(maxIter,K);
    
    % Initilize clusters
    %if strcmp(init,'uniform')
        X_range = [min(X(:)),max(X(:))];
        C = unifrnd(X_range(1),X_range(2),p,K); 
    %elseif strcmp(init,'plus')
        %C = pdfc_diametrical_clustering_plusplus(X,K,stream);
   % end
    
    it = 0;
    while true
        it = it + 1; % E-step, similarity between all samples and all centroids
        
        dis2 = zeros(n,K);
        
        for ii = 1:n
            for jj = 1:K
                dis2(ii,jj) = distancefun(C(:,jj)',X(ii,:));    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        end
        
        [mindis,X_part] = min(dis2,[],2); % minimum value for each row, + indexes - so, what is the closest centroid
        objective(it) = mean(mindis);
        
        for k = 1:K
            partsum(it,k) = sum(X_part==k); % they put this sum of assignations as check for convergence...
        end
        
        if it>1
            if isequal(partsum(it,:),partsum(it-1,:))|| (it == maxIter)
                C_final(:,:,repl) = C; % capire dimensionalita
                objective_final(:,repl) = objective(end);
                X_part_final(:,repl) = X_part;
                break
            end
        end
        
        
        % M-step, update centroids according to allocated samples
        for k = 1:K
            idx_k = X_part==k;
            C(:,k) = mean(X(idx_k,:),1)'; % if there are no assignments, this does not work...let me try to fix
            % I fix this way: if no assignment, C will be another random
            % number
            if sum(idx_k)==0
                C(:,k) = unifrnd(X_range(1),X_range(2),p,1);
            end
        end
        
        
    end
end


% Output result from best replicate
[~,idx_obj] = min(objective_final,[],2);
Obj_out = objective_final(:,idx_obj);
C_out = C_final(:,:,idx_obj)';
idx_out = X_part_final(:,idx_obj);

end