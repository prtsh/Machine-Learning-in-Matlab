function [w,b, obj, cvErrs] = ridgeReg(X,y, lambda)
% Ridge regression implementation
% inputs
% X :  k x n, k features, n data points
% y :  n x 1, n labels
% lambda : 1x1, constant, regularization vector
% outputs
% w : K x 1 vector
% b : scalar for bias
% obj : objective function value
%cvErrs : n x 1 vector for cv errors, cvErrs(i) is the leave-one-out error
%         for removing the ith training data
X = X';
[row, col] = size(X);
Ones_Ncol = ones(1, col);
X_bar = [X;Ones_Ncol];
X_bar_T = X_bar';
I_bar = eye(row+1);
I_bar(:,row+1) = 0;
C = X_bar*X_bar_T + lambda*I_bar;
d = X_bar*y;

W_bar = C\d; 
%Weight vector, K x 1
w = W_bar(1:row, :); 
%scalar value for the bias term
b = W_bar(row+1,1); 
%objective function
obj = (X_bar'*W_bar - y)'*(X_bar'*W_bar - y) + lambda*W_bar'*I_bar*W_bar;
%cvErrs, inefficient method, LOOCV
cvErrs = ones(col,1);
cvErrs = cvErrs.*realmax;

%cverrs is super slow, disable if not required
enable = 0;  %enable cvErrs
if enable == 1
for i = 1:col %for ach sample, remove it and train the data
    x_i = X(:,i);
    y_i = y(i,:);
    x_i_bar = X_bar(:, 1);
    %equation 6 in the assignments
    cvErrs(i,1) = (w'*x_i - y_i)/(1 - (x_i_bar'/C)*x_i_bar);
end
end

