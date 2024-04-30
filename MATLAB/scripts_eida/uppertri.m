function out = uppertri(matrix)

% this function outputs the upper triangular part of a matrix as a vector


out = triu(matrix);
out = out(:);
out = out(out~=0);

end

