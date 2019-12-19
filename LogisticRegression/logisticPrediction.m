
%@function : A utility function to find the predicted labels 
%@params   : theta, parameter obtained from training
%            X_input, validation samples
%@return   : Predicted label matrix
function Y_pred = logisticPrediction(theta, X_input)
    X = normalize(X_input, 1);
    samples = size(X,1);
    Y_pred = zeros(samples, 1);
    Ones_Ncol = ones(samples, 1);
    X_bar = [X,Ones_Ncol];
    tolerance = 1.0e-9;
    Loss1 = 0; Loss2 = 0; Loss3 = 0; Loss4 = 0; Loss_total = 0;
    for i = 1:samples
        X_i = X_bar(i, :)';
        norm = 1 + exp(theta(:, 1)'*X_i) + exp(theta(:, 2)'*X_i) ...
                         + exp(theta(:, 3)'*X_i);
        PY_XiC1  = exp(theta(:, 1)'*X_i)/(norm);
        PY_XiC2  = exp(theta(:, 2)'*X_i)/(norm);
        PY_XiC3  = exp(theta(:, 3)'*X_i)/(norm);
        PY_XiC4  = 1/(norm);
        
        Loss1 = Loss1 + log(PY_XiC1);
        Loss2 = Loss2 + log(PY_XiC2);
        Loss3 = Loss3 + log(PY_XiC3);
        Loss4 = Loss4 + log(PY_XiC4);
        
        
        predicted = max([PY_XiC1, PY_XiC2, PY_XiC3, PY_XiC4]);
        if abs(predicted - PY_XiC1) <= tolerance
            label = 1;
        elseif abs(predicted - PY_XiC2) <= tolerance
            label = 2;
        elseif abs(predicted - PY_XiC3) <= tolerance
            label = 3;
        else
            label = 4;
        end 
        
        Y_pred(i,1) = label;
    end
    Loss_total = -(Loss1 + Loss2 + Loss3 + Loss4)/samples;
        disp(Loss_total);
    writematrix(Y_pred, 'Test_Labels.csv')
end
