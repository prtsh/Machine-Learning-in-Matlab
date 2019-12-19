%@function: compute exp-chi-square kernel
%@return: transormed train and test data
function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)
    n = size(trainD,1);
    trainK = ones(size(trainD,1),size( trainD,1));
    testK  = ones(size(testD,1), size(trainD,1));

    %for training data
    for i=1:n
     x = trainD(i,:)'; 
     for j=1:n  
         y = trainD(j,:)';
         kv = exp((-1/gamma)*(sum(((x - y).^2)./(x + y + 0.000001))));%chi square kernel
         trainK(i,j) = kv;
     end
    end
    trainK = [(1:n)', trainK];

    %for test data
    m = size(testD, 1);
    for i=1:m
    x = testD(i,:)'; 
      for j=1:n
        y = trainD(j, :)';      
        kv = exp((-1/gamma)*(sum(((x - y).^2)./(x + y + 0.000001))));%chi square kernel
        testK(i, j) = kv;
      end
    end
    testK = [(1:m)', testK];
end