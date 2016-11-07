function [sgm] = sigmoid(x)
%Sigmoid function from R->R
%Input: x in R
%Output: 1/(1+exp(-x))

sgm = 1./(1+exp(-x));

end

