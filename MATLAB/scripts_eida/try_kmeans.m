% try if kmeans works

clear,close

k = 5; 

data = [ones(20,10); 2*ones(20,10);3*ones(20,10);4*ones(20,10);5*ones(20,10)];


[idx,C,~] = kmeans_custom_distance(data,k,200,50,[],true,@myeuclidean);


plot(idx);


function dist = myeuclidean(x,y)

dist = norm(x-y,2);

end