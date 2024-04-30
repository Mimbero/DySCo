function C = generate_observable(N)
% this function generates a symmetric, positive semidefinite matrix, given
% the input size. We call it observables because in quantum mechanics this
% would be an observable

% buid the random matrix  by summing lambda_i*vi*vi' in order to obtain a 
% symmetric positive semidefinite matrix

C = zeros(N,N);

for i=1:N
 v = randn(N,1); % random eigenvector
 lambda = unifrnd(0,1); %random positive eigenvalue NOTA forse in futuro fai gli autovalori pesati per essere realistico
 C = C + lambda*(v*v');
end

end


