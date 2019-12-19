function [X_new, y_new] = randomTest(X, y, R)
% Utility function, For quick testing, training etc.on a subset of X and labels
% randomly select R test samples from the input data for testing
% return new test input and new label
% this function can be used to run like 500 samples instead all 4000
% see the results

X_t = X';
X_t = [X_t, y];
[N,K] = size(X_t);
k = randperm(N);
X_new_bar= X_t(k(1:R),:);
y_new = X_new_bar(:,K);
X_new_bar(:,K) = []; 
X_new = X_new_bar; %new [Samples, label], subset of [X, y]
end

