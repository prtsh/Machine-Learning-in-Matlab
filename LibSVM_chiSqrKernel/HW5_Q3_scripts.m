clc
clear
addpath '/Users/pratyushkr/Desktop/cse512/hw_5/libsvm-3.24/matlab'
scales = [8, 16, 32, 64];
normH = 16;
normW = 16;
[trIds, trLbs] = ml_load('../bigbangtheory/train.mat',  'imIds', 'lbs');             
tstIds = ml_load('../bigbangtheory/test.mat', 'imIds'); 
%bowCs = HW5_BoW.learnDictionary(scales, normH, normW);
bowCs   = ml_load('../src/bowCs.mat', 'bowCs'); %saved from a previous K-mean training

%trD  = HW5_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
%tstD = HW5_BoW.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);
trD   = ml_load('../src/trD_compfeatures.mat', 'trD'); %saved from a previous feature calculation
tstD   = ml_load('../src/tstD_compfeatures.mat', 'tstD'); %saved from a previous feature calculation
trD  = trD';
tstD = tstD';

%{
svmtrain, for reference purpose
Usage: model = svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options');
libsvm_options:
-s svm_type : set type of SVM (default 0)
	0 -- C-SVC		(multi-class classification)
	1 -- nu-SVC		(multi-class classification)
	2 -- one-class SVM
	3 -- epsilon-SVR	(regression)
	4 -- nu-SVR		(regression)
-t kernel_type : set type of kernel function (default 2)
	0 -- linear: u'*v
	1 -- polynomial: (gamma*u'*v + coef0)^degree
	2 -- radial basis function: exp(-gamma*|u-v|^2)
	3 -- sigmoid: tanh(gamma*u'*v + coef0)
	4 -- precomputed kernel (kernel values in training_instance_matrix)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
-v n: n-fold cross validation mode
-q : quiet mode (no outputs)
%}

%3.4.2
%%{
%cv_accuracy = svmtrain(trLbs, trD, '-v 5');
%%}

%3.4.3
%{
%macro level grid search and tuning
C = [1, 10, 25, 50, 100, 200];
gamma = [1, 10, 25, 50, 100, 200];
            
for i = 1:6
    for j = 1:6
        opt = sprintf(' -m 1024 -q -v 5 -c %d -g %d', C(i), gamma(j));
        cv_accuracy = svmtrain(trLbs, trD, opt);
        if (cv_accuracy > max_accuracy)
            max_accuracy = cv_accuracy;
            c = C(i);
            g = gamma(j);
        end
        fprintf('(Max accuracy: %f gamma: %f, C: %f \n',max_accuracy, g, c);
    end
end
%}

%3.4.5
%{
% cmpExpX2Kernel 
% macro level grid search and tuning for chi-square kernel

C = [1, 10, 25, 50, 100, 200, 400, 800];
gamma = [0.1, 1, 2, 4, 8, 16, 32];

max_accuracy = 0;
for j = 1:9
    [trainK, testK] = cmpExpX2Kernel(trD, tstD, gamma(j));
    for i = 1:6
        opt = sprintf('-m 1024 -q -v 5 -c %d -g %d', C(i), gamma(j));
        cv_accuracy = svmtrain(trLbs, trD, opt);
         if (cv_accuracy > max_accuracy)
             max_accuracy = cv_accuracy;
             c = C(i);
             g = gamma(j);
         end
    end
    fprintf('Current Max accuracy: %f gamma: %f, C: %f \n',max_accuracy, g, c);
end
%}

%{
% micro level grid search and 
% fine tuning for chi-square kernel
%C = [ 24, 26, 28, 30];
%gamma = [2.6];

C = [3125, 3130,  3140, 3150];
gamma = [2.6];

max_accuracy = 0;
for j = 1:length(gamma)
    [trainK, testK] = cmpExpX2Kernel(trD, tstD, gamma(j));
    for i = 1:length(C)
        fprintf('training for gamma: %f, C: %f \n', gamma(j), C(i));
        opt = sprintf('-m 1024 -q -v 5 -c %d -g %d', C(i), gamma(j));
        cv_accuracy = svmtrain(trLbs, trD, opt);
        if (cv_accuracy > max_accuracy)
            max_accuracy = cv_accuracy;
            c = C(i);
            g = gamma(j);
        end
        fprintf('Current Max accuracy: %f gamma: %f, C: %f \n',max_accuracy, g, c);
    end
end

%}

%3.4.6
%%{
%final prediction with tuned parameters
gamma= 2.6; C = 3200;
Images = ml_load('../bigbangtheory/test.mat', 'imIds');
[trainK, testK] = cmpExpX2Kernel(trD, tstD, gamma);
opt = sprintf('-m 1024 -t 4 -q -c %d', C);
svm = svmtrain(trLbs, trainK, opt);

n = size(testK,1);
tstLbs = rand(n, 1);
[Prediction] = svmpredict(tstLbs, testK, svm);

ImgId = 1:1:length(Images);
predicted_data = [ImgId', Prediction];
csvwrite('predTestLabels.csv', predicted_data);
%%}