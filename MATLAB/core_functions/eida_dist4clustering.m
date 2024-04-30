function D = eida_dist4clustering(ZI,ZJ)

% this function just takes the eida distance, but does it in the way that
% you can input it to a clustering algorithm 

% note that I'm setting all these parameters here because i don't know how
% to pass the distance to the function. i know we shoul change it

N = 10; % porcatella da cambiare 
n_eigen = 9;
what_distance = 2;

n_elements = size(ZJ,1); % number of elements in ZJ to do the distance
D = nan(n_elements,1);

M1 = reshape(ZI,N,n_eigen); % rebuild the matrix as it's required for eida distances

for i=1:n_elements
    
    M2 = reshape(ZJ(i,:),N,n_eigen);
    D(i) = real(eida_distance(M1,M2,what_distance));
end


end