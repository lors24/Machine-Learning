function [a,z] = feedforward(a,W,b,f)
%Input:
%a: input
%W: weights
%b : bias
%f : function 

z = W'*a+b;
a = feval(f,z);


end

