function D = trace_complex(v1,ZJ)


m2 = size(ZJ,1);
D = nan(m2,1);


for i=1:m2
    
    v2 = ZJ(i,:);
    D(i) = sqrt(1-abs(v1*v2')^2);
end

end

