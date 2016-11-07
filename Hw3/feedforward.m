function [out] = feedforward(a,W,b,f)
%Input:
%a: input
%W: weights
%b : bias
%f : function 

z = W'*a+b;
out = feval(f,z);


end

