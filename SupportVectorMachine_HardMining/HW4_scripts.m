clear
clc

%{
%problem 2.4
[alpha, fval, validindex, sv] = svmduallinear(trD, trLb, C, 'linear');
[w, b]  = svmprimal(X, Y, alpha, validindex);
[accuracy, confmatrix] = validateSVM(w, b, valD, valLb);
%}


%%{
%problem 2.6
%multiclass SVM using one-vs-all training algorithm 
%trD   = csvread(SVM_HW4_Q3.dataDir + "/cse512hw4/"+'train'+"_Features.csv",0,1);
%trLb  = csvread(SVM_HW4_Q3.dataDir + "/cse512hw4/"+'train'+"_Labels.csv",1,1);
%valD  = csvread(SVM_HW4_Q3.dataDir + "/cse512hw4/"+'val'+"_Features.csv",0,1);
%valLb = csvread(SVM_HW4_Q3.dataDir + "/cse512hw4/"+'val'+"_Labels.csv",1,1);
%testD = csvread(SVM_HW4_Q3.dataDir + "/cse512hw4/"+'test'+"_Features.csv",0,1);
%[Y_pred, w_k, b_k] = SVM_HW4_Q3.multiSVM(trD', trLb, 10.0, 'linear');
%[Y_final, accuracy, confmatrix] = SVM_HW4_Q3.scoreMultiSVM(Y_pred, trLb);
%[Y_pred] = SVM_HW4_Q3.predictMultiSVM(w_k,b_k, testD);
%csvwrite("Test_Labels.csv",Y_pred);
%%}



%for problem 3.4.1, 
%generate AP/pr-rec curve on val data
%{
load("./data/q2_1_data.mat")
run('vlfeat-0.9.21/toolbox/vl_setup.m');
[trD_, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
trD = normalize(trD_,2, 'norm');
[alpha, fval, validindex, sv] = SVM_HW4_Q3.svmdual(trD, trLb, 0.3, 'linear');
[w,b] = SVM_HW4_Q3.svmprimal(trD, trLb, alpha, validindex);
HW4_Utils.genRsltFile(w, b, 'val', 'HW4_Q3_valdataresult');
[ap, prec, rec] = HW4_Utils.cmpAP('HW4_Q3_valdataresult', 'val');
%}


%for problem 3.4.2. 3.4.3, 3.4.4
run('vlfeat-0.9.21/toolbox/vl_setup.m');
SVM_HW4_Q3.hardNegativeMiningSVM();


