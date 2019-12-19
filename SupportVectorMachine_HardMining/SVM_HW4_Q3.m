
classdef SVM_HW4_Q3
    %@class:  solves SVM objective in the dual form( the langragian derivative
    %         of the primal form, implements one-vs-all multisvm
    %         includes related methods for training, validation and prediction
     properties (Constant)        
        dataDir = './data';
        %SVM tuning params
        epsilon = 1e-6;
        classes = 4
        gamma = 1.0
        polynom = 2
        %hardmining params
        negMineIter = 10
        overlapThresh = 33
        alphaThresh = 0.05
        hardNegCount = 1000
        C = 10
     end
     
     methods (Static)
         %useful for loading cse512hw4 data
         %for problem 2.6
         function [trD, trLb]  = readdata(name)
             trD  = csvread(SVM_HW4_Q3.dataDir + "/cse512hw4/"+name+"_Features.csv",0,1);
             trLb = csvread(SVM_HW4_Q3.dataDir + "/cse512hw4/"+name+"_Labels.csv",1,1);
         end
         
         %utility to plot graphs
         function plot(fvals, APs)             
             x = 1:1:SVM_HW4_Q3.negMineIter;
             
             tiledlayout(2,1)
             ax1 = nexttile;
             plot(ax1,x,fvals)
             title(ax1,'fvals')
             ylabel(ax1,'fvals(objective func. value)')
             xlabel(ax1, 'iterations')
             
             ax2 = nexttile;
             plot(ax2,x,APs)
             title(ax2,'Precision')
             ylabel(ax2,'AP')      
             xlabel(ax1, 'iterations')
         end
         
         %linear kernel for SVM
         function Y=Linear(X1,X2)
            Y=zeros(size(X1,1),size(X2,1));
            for i=1:size(X1,1)
                for j=1:size(X2,1)
                    Y(i,j)=dot(X1(i,:),X2(j,:));
                end
            end
         end
        
         %RBF kernel for SVM
         function Y=RBF(X1,X2)
            Y=zeros(size(X1,1),size(X2,1));
            for i=1:size(X1,1)
                for j=1:size(X2,1)
                    Y(i,j)=exp(-SVM_HW4_Q3 .gamma*norm(X1(i,:)-X2(j,:))^2);
                end
            end
         end
         
         %polynomial kernel
         function Y=Polynomial(X1,X2)
            Y=zeros(size(X1,1),size(X2,1));
            for i=1:size(X1,1)
                for j=1:size(X2,1)
                    Y(i,j)=(1+dot(X1(i,:),X2(j,:))).^SVM_HW4_Q3.polynom;
                end
            end
         end
         
         %@function: solves the dual quadratic svm problem, 
         %@input: takes C as input(upper bound of alpha)
         %        takes the input training data and training labels           
         %@return: return the aplha, objecive value and supportvectr count
         function [alpha, fval, validindex, sv] = svmdual(trD, trLb, C, kernel)
             
             switch kernel
                case 'linear'
                    Kernel=SVM_HW4_Q3.Linear(trD',trD');
                case 'polynomial'
                    Kernel=SVM_HW4_Q3.Polynomial(trD',trD');
                case 'RBF'
                    Kernel=SVM_HW4_Q3.RBF(trD',trD');
                case 'Sigmoid' %currently not implemented
                    Kernel=SVM_HW4_Q3.Sigmoid(trD',trD');
             end

             %X = X';
             samples = size(trD, 2);
             Y = trLb';
             H = diag(Y)*Kernel*diag(Y);
             f = -ones(samples,1);
             Aeq = Y; 
             beq = 0;
             A = [];
             b = [];
             lb = zeros(samples,1);  %for aplha it's 0
             ub = C * ones(samples,1); %for alpha it's C
             options = optimoptions('quadprog',...
                    'Algorithm','interior-point-convex','Display','off');
             [alpha, fval] = quadprog(H,f,A,b,Aeq,beq,lb,ub, [], options);      
             nearzero = alpha < SVM_HW4_Q3.epsilon; 
             nearC = (alpha > (C - SVM_HW4_Q3.epsilon));
             if sum(nearzero) >= 1
                 alpha(nearzero) = 0; 
             end            
             if sum(nearC) >= 1
                 alpha(nearC) = C; 
             end            
             sv = length(find(alpha ~= 0));                       
             validindex = find(0<alpha & alpha<=C);
         end
         
         %@function: takes the alpha from the svmdual as input
         %@return: the hyperplane parametrs (w,b) separating data
         %@note:  only linear is implemented
         function [w, b]  = svmprimal(trD, trLb, alpha, validindex)
             w = 0;
             Y = trLb';
             alpha = alpha';
             validindex = validindex';
             for j = validindex %fr each support vector
                w = w+alpha(j)*Y(j)*trD(:,j);
             end
            b=Y(validindex)-w'*trD(:,validindex);
            b = mean(b);
         end
         
         %@function: validate label using previously trained SVM, w and b
         %@return: report the accuracy, the objective value of SVM, 
         %         the number of support vectors, and the confusion matrix.
         %@note:  only linear is implemented
         function [accuracy, confmatrix] = validateSVM(w, b, valD, valLb)
             %f(x) = w'x + b
             f = w'*valD + b;
             Y_pred = sign(f);
             accuracy = 0.0;
             samples = size(Y_pred,2);
             for i = 1:samples
                 if Y_pred(i) == valLb(i)
                     accuracy = accuracy+1;
                 end
             end
             accuracy = accuracy/samples;
             confmatrix = confusionmat(valLb, Y_pred);
             %plot cofuion matrix
             confusionchart(confmatrix);
         end
         
         %@function: a multiclass SVM using one-vs-all training algorithm 
         %@note:  only linear is implemented
         function [Y_pred, w_k, b_k] = multiSVM(trD, trLb, C, kernel)
             % normalize across column, normalize all data for a given
             % feature, default is z_scalar, try different types
             X = normalize(trD, 1); 
             %X = trD;
             %build models one vs all, for each class
             Y_pred = ones(size(X,2), SVM_HW4_Q3.classes);
             iteration = 1;
             b_k = zeros(1, SVM_HW4_Q3.classes); %a "b"for each class
             w_k = zeros(size(trD,1), SVM_HW4_Q3.classes);
             for k=1:SVM_HW4_Q3.classes
                 fprintf('\nIteration %d, training classifier: %d\n', iteration, k); 
                 iteration = iteration+1;
                 x_mat = trD;
                 y_vec = trLb;
                 m_class = (y_vec == k);
                 y_vec(m_class) = 1;
                 y_vec(~m_class) = -1/(SVM_HW4_Q3.classes-1); %negative class target              
                 [alpha,~,validindex,~] = SVM_HW4_Q3.svmdual(trD, y_vec, C, kernel);
                 [w, b]  = SVM_HW4_Q3.svmprimal(trD, y_vec, alpha, validindex);
                 %disp(b)
                 b_k(:,k) = b;
                 w_k(:,k) = w;
                 
                 % make predictions
                 Y_pred(:,k) = SVM_HW4_Q3.predictSVM(w,b, x_mat);
             end
         end
         
         %@function: validate a trained multiSVM and return the score
         %@note:  only linear is implemented
         function [Y_final, accuracy, confmatrix] = scoreMultiSVM(Y_pred, Y_val)
             samples = size(Y_pred,1);
             Y_final = ones(length(Y_val),1);
             accuracy = 0;
             for i = 1:samples
                 %check for prediction in Y_pred, obtained from multiSVM()
                 if Y_pred(i, Y_val(i)) == 1 
                     Y_final(i) = Y_val(i);
                     accuracy = accuracy+1;
                 end
             end
             %calualte accuracy and confmatrix
             accuracy = accuracy/samples;
             confmatrix = confusionmat(Y_val, Y_final);
             confusionchart(confmatrix);
         end
         
         %@function: predict the Y labels to upload, using the precomuted
         %           w, and b values,
         %@ parameters: w and b are obtained from training, see multiSVM()
         %@ return: final labels, used on the test data
         %@note:  only linear is implemented
         function [Y_test] = predictSVM(w, b, X_test)
             f = w'*X_test + b;
             Y_test = sign(f);
         end
         
         %@function: predict the Y labels to upload, using the precomuted
         %           w_k, and b_k values,
         %@ parameters: w_k and b_k are obtained from training, see multiSVM()
         %@ return: final labels
         function [Y_pred] = predictMultiSVM(w_k,b_k, X_test)
             X = normalize(X_test, 1);
             samples = size(X_test, 1);
             %build models one vs all, for each class
             Y_pred_k = ones(size(X,1), SVM_HW4_Q3.classes);
             Y_pred = ones(size(X_test,1),1);
             iteration = 1;
             for k=1:SVM_HW4_Q3.classes
                 fprintf('\nIteration %d, Predicting classifier: %d\n', iteration, k); 
                 iteration = iteration+1;
                 x_mat = X';
                 % make predictions
                 Y_pred_k(:,k) = SVM_HW4_Q3.predictSVM(w_k(:,k),b_k(:,k), x_mat);
             end
             
             for i = 1:samples
                 %check for prediction in Y_pred, obtained from multiSVM()
                 for k = 1:SVM_HW4_Q3.classes
                    if Y_pred_k(i,k) == 1 
                        Y_pred(i) = k;
                        break;
                    end
                 end
             end
         end
         
         %@function: predict the Y labels to upload, using the precomuted
         %           w_k, and b_k values,
         %@ parameters: w_k and b_k are obtained from training, see multiSVM()
         % readings - https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj5/html/agartia3/index.html
         function hardNegativeMiningSVM()
             load("./data/trainAnno.mat", 'ubAnno'); %for upper body annotations on all trained images
             [trD, trLb, ~, ~, ~, ~] = HW4_Utils.getPosAndRandomNeg();
             %first training 
             [alpha, ~, validindex, ~] = SVM_HW4_Q3.svmdual(trD, trLb, SVM_HW4_Q3.C, 'linear');
             [w ,b] = SVM_HW4_Q3.svmprimal(trD, trLb, alpha, validindex);
             fvals = ones(SVM_HW4_Q3.negMineIter,1);
             APs = ones(SVM_HW4_Q3.negMineIter,1);
             temp = trD';
             PosD = temp(trLb==1,:)';
             NegD = temp(trLb==-1,:)';
             %number of hard mining iterations
             for iter = 1:SVM_HW4_Q3.negMineIter
                fprintf('Hard negative mining Iteration: %d\n', iter);
                HW4_Utils.genRsltFile(w, b, "train", "HW4_Q3_rects");
                load("HW4_Q3_rects.mat", 'rects');
                B = SVM_HW4_Q3.findHardNegs(rects, ubAnno); %hard examples
                A = temp(alpha < SVM_HW4_Q3.alphaThresh,:)'; 
                A = A'; NegD = NegD'; B = B';
                NegD = NegD(~ismember(NegD, A, 'rows'),:); %remove non support and union with B
                NegD = [NegD; B];
                A = A'; NegD = NegD'; B = B';
                temp = [PosD'; NegD'];
                trD = temp';
                trLb = [ones(size(PosD, 2), 1);-1 * ones(size(NegD, 2), 1)];
                [alpha, fval, validindex, ~] = SVM_HW4_Q3.svmdual(trD, trLb, SVM_HW4_Q3.C, 'linear');
                [w ,b] = SVM_HW4_Q3.svmprimal(trD, trLb, alpha, validindex); 
                fvals(iter,:) = fval;
                HW4_Utils.genRsltFile(w, b, "val", "HW4_Q_3_4");
                [ap, ~, ~] = HW4_Utils.cmpAP("HW4_Q_3_4", "val");
                APs(iter,:) = ap;
             end
             HW4_Utils.genRsltFile(w, b, "test", "112675752");
             SVM_HW4_Q3.plot(fvals, APs);
         end
             
         %@function: mine hard negative overlaps
         function hard = findHardNegs(rects, ubAnno)
                hard = [];
                for i = 1:length(rects)
                    if size(hard, 2) > SVM_HW4_Q3.hardNegCount
                       break;
                    end
                    im = imread(sprintf('%s/%sIms/%04d.jpg', HW4_Utils.dataDir, "train", i));
                    imH = size(im,1);   imW = size(im,2);                                      
                    ubs = ubAnno{i};
                    for box = 1:length(rects{i})   
                        if rects{i}(3,box) > imW || rects{i}(4,box) > imH %if the region is beyond the image boundary
                            continue;
                        end
                        if rects{i}(5,box) > 0 %score value of a rect region, ignore positive scores
                            continue;
                        end
                        %overlaps = [];
                        for j = 1:size(ubs, 2)
                            overlap = HW4_Utils.rectOverlap(rects{i}, ubs(:, j));
                            if overlap(i) < SVM_HW4_Q3.overlapThresh %not a negative enough sample
                                coord1 = int16(rects{i}(2, box)); coord2 = int16(rects{i}(4, box));
                                coord3 = int16(rects{i}(1, box)); coord4 = int16(rects{i}(3, box));
                                imReg = im(coord1:coord2, coord3:coord4, :);
                                imReg = imresize(imReg, HW4_Utils.normImSz);
                                hog = (HW4_Utils.cmpFeat(rgb2gray(imReg)));
                                hog = hog/norm(hog);
                                hard = [hard, hog];
                            end
                        end
                    end
               end
         end
     end
end