function D = norm2_distance(ZI,ZJ)

% This function rebuilds the full matrix from the upper triangular
% representation and then computes the trace distance 

dim = 10; % porcatella da cambiare 

m2 = size(ZJ,1);
D = nan(m2,1);

M1 = unpack_matrix(ZI,dim)/dim;

for i=1:m2
    
    M2 = unpack_matrix(ZJ(i,:),dim)/dim;

    rho = M1-M2;
    D(i) = norm(rho,2);
end




%eigs = eig(rho);
%dist = sum(abs(eigs))/dim;

end