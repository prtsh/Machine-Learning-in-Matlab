
%@function : find the validation loss 
%@params   : theta, parameter obtained from training
%            X_input, validation samples
%@return   : Validation Loss
function L_validation_theta = logisticValidation(theta, X_input)
    X = normalize(X_input, 1);
    samples = size(X,1);
    Ones_Ncol = ones(samples, 1);
    X_bar = [X,Ones_Ncol];
    Loss1 = 0; Loss2 = 0; Loss3 = 0; Loss4 = 0;
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
        
    end
    Loss_total = -(Loss1 + Loss2 + Loss3 + Loss4)/samples;
    L_validation_theta = Loss_total;
    disp(Loss_total);
end