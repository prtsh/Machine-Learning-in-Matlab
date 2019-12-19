function [rmse, sse, pred_y] = RMSE(X,y,w,b)
%@return: root mean square error and sum of square or error

%RMSE error
%Solve Ridge to predict the number of points a Wine will receive. Run Ridge on the training
%set, with λ = 0.01, 0.1, 1, 10, 100, 1000. At each solution, record the root-mean-squared-error (RMSE)
%on training, validation and leave-one-out-cross-validation data
 X = X';
 N = size(X,2);
 pred_y = (w'*X + b);
 err_y = pred_y - y'; %error: predicted - original
 err_y = err_y.^2; %square of errors
 sse = sum(err_y); %sum of square of errors
 rmse = (sse/N).^0.5;
 
end

