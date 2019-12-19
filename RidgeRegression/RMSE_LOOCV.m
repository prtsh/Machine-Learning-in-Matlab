function [rmse,sse] = RMSEcvErr_LOOCV(cvErrs)
 N = size(cvErrs,1); 
 err_y = repmat(cvErrs, 1);
 err_y = err_y.^2; %square of errors
 sse = sum(err_y); %sum of square of errors
 rmse = (sse/N).^0.5; %root of mean of square of errors
end

