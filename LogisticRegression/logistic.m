
%Logisctic regression with good ol' SGD and batching
function theta = logistic(X_input, Y, m, eta_0, eta_1, maxepoch, delta)
    %normalize each column, either range or z-normalization, (x-mean)/std_dev
    X = normalize(X_input, 1);
    [n, features] = size(X);
    Ones_Ncol = ones(n, 1);
    X_bar = [X,Ones_Ncol];
    Y_bar = Y;
    %combine sample vector with label vector
    %in the form [X, Y]
    XY = [X_bar, Y_bar];
    
    theta = rand(features+1, 4);
    %theta for each epoch, saving upto 100 epochs
    theta_epoch = zeros(features+1, 4, 100);
    %loss for each epoch, saving upto 100 epochs
    Loss_epoch = zeros(100, 1);
    Epoch_count = 0;
    batches = floor(n/m); 
    Loss_theta_old = intmax;
    for epoch = 1:maxepoch
        disp(Loss_theta_old);
        Epoch_count = Epoch_count+1;
        eta = eta_0/(eta_1 + epoch);
        %permute the input data [X,Y]
        random_XY = XY(randperm(size(XY, 1)), :);
        X_epoch = random_XY(:,1:end-1); 
        Y_epoch = random_XY(:,end);
        offset = 0;
       
        Loss1 = 0; Loss2 = 0; Loss3 = 0; Loss4 = 0;
        for batch = 1:batches
            sum_gradient1 = zeros(features+1,1);
            sum_gradient2 = zeros(features+1,1);
            sum_gradient3 = zeros(features+1,1);
            sum_gradient4 = zeros(features+1,1);
            
            %for each of the X_i in this batch, process the entire batch
            for i = batch*offset+1 : batch*offset + m      
                if i > n
                    break;
                end
                X_i = X_epoch(i,:)';
                Y_ground = Y_epoch(i,:);
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
                
                %equation 8
                loss_gradient_1 = ((Y_ground == 1) - PY_XiC1)*X_i;
                loss_gradient_2 = ((Y_ground == 2) - PY_XiC2)*X_i;
                loss_gradient_3 = ((Y_ground == 3) - PY_XiC3)*X_i;
                loss_gradient_4 = ((Y_ground == 4) - PY_XiC4)*X_i;
                
                sum_gradient1 = sum_gradient1 + loss_gradient_1;
                sum_gradient2 = sum_gradient2 + loss_gradient_2;
                sum_gradient3 = sum_gradient3 + loss_gradient_3;
                sum_gradient4 = sum_gradient4 + loss_gradient_4;
                %now out of the batch for loop
            end
            
            offset = offset + 1;
            %assignment equation 7
            sum_gradient1 = -sum_gradient1/m;
            sum_gradient2 = -sum_gradient2/m;
            sum_gradient3 = -sum_gradient3/m;
            sum_gradient4 = -sum_gradient4/m;
            
            % update theta,the gradient descent
            % learning rate is eta, 
            % gradient for class c is sum_gradientc
            % assignment equation 5
            theta_old = repmat(theta,1);
            theta(:, 1) = theta_old(:,1) - eta*sum_gradient1;
            theta(:, 2) = theta_old(:,2) - eta*sum_gradient2;
            theta(:, 3) = theta_old(:,3) - eta*sum_gradient3;
            theta(:, 4) = theta_old(:,4) - eta*sum_gradient4;
        end
        
         % terminating condition, assignment equation 4
        Loss_theta_new = -(Loss1 + Loss2 + Loss3 + Loss4)/n;
        
        
        if Loss_theta_new > (1-delta)* Loss_theta_old
            break;
        else
            Loss_theta_old = Loss_theta_new;
        end
        Loss_epoch(epoch,1) = Loss_theta_new; 
        writematrix(Loss_epoch, 'loss_epoch.csv')
        theta_epoch(:,:,epoch) = theta;
        writematrix(theta_epoch, 'theta_epoch.csv')
    end
end
%Outputs: theta
    