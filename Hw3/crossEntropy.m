function [loss] = crossEntropy(y,t)
%y: Predicted probability from Softmax function
%t: Target value

loss = sum(sum(-t.*log(y)));

end

