function D = eig_distance(ZI,ZJ)

% This function rebuilds the full matrix from the upper triangular
% representation and then computes the trace distance 

dim = 10; % porcatella

m2 = size(ZJ,1);
D = nan(m2,1);

M1 = unpack_matrix(ZI,dim)/dim;
[v1,~] = eigs(M1,1);

for i=1:m2
    
    M2 = unpack_matrix(ZJ(i,:),dim)/dim;
    [v2,~] = eigs(M2,1);
    D(i) = pdist([v1';v2'],'cosine');
end

end