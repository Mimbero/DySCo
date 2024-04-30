% new random things on matlab

% try if the main formula works

vectors = randn(20,10);
weights = unifrnd(0,2,10,1);

matrix = zeros(20,20);

for i=1:10
    
    matrix = matrix+ weights(i)*vectors(:,i)*vectors(:,i)';
    
end

[v,d] = eigs(matrix,10);

% now try my formula

scalars = vectors'*vectors;

for i=1:10
    
    scalars(i,:) = weights(i)*scalars(i,:);
    
end

[v1,d_my_formula] = eigs(scalars,10);

v_my_formula =vectors*v1;

for i=1:10
    
    v_my_formula(:,i) = v_my_formula(:,i)/norm(v_my_formula(:,i));
    v(:,i) = v(:,i)/norm(v(:,i));
    
end

%% now try von neumann vs zip

input_data = readmatrix('prova_animale.txt');

von_neumanns = zeros(1,330);
zips = zeros(1,330);

for i=1:330
    
    matrix = input_data(i:i+20,:);
    
    % Save the matrix as a binary file
    save('matrix.bin', 'matrix', '-v6');
    % Compress the binary file
    system('gzip matrix.bin');
    % Calculate the number of bits in the compressed file
    fileInfo = dir('matrix.bin.gz');
    compressedFileSize = fileInfo.bytes;
    bits = compressedFileSize * 8;
    % Delete the compressed file
    delete('matrix.bin.gz');
    
    zips(i) = bits;
    
    
    % now try with von_neumann
    
    density = matrix'*matrix;
    [~,lambda] = eigs(density,10);
    lambdas = diag(lambda);
    pis = lambdas/sum(lambdas);
    
    von_neumanns(i) = -sum(pis.*log(pis));
end

scatter(von_neumanns,zips)
corr(von_neumanns',zips')


