function matrix = unpack_matrix(m,dim)

% this fcn gives you a matrix from the upper triangle vector m


matrix = ones(dim,dim);
ind = 1;

for column=1:dim
    for row=1:column
        matrix(row,column) = m(ind);
        matrix(column,row) = m(ind);
        ind = ind+1;
    end
end

end
