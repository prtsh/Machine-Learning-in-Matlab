%@function : Find accuracy
%@params   : predicted labels, ground truth labels
%@return   : accuracy score
function score = accuracy(Y_predicted, Y_labels)
    samples = size(Y_predicted,1);
    total_correct = 0;
    for i = 1:samples
        total_correct = total_correct + (Y_predicted(i,:) == Y_labels(i,:));
    end
    %accuracy = (total_correct/total_samples)
    score = total_correct/samples;
end