function D = cosine_distance(ZI,ZJ)

% This function rebuilds the full matrix from the upper triangular
% representation and then computes the trace distance 

m2 = size(ZJ,1);
D = nan(m2,1);

for i=1:m2
    D(i) = pdist([ZI;ZJ(i,:)],'cosine');
end

end