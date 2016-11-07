function [loss] = crossEntropy(y,t)
%y: Predicted probability from Softmax function
%t: Target value

loss = -t*log(y);

end

