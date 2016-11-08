function [delta] = backprop(d,z,W)
%Backpropagation
%Input:
%d = delta
%z 
%W = weights

[m ~] = size(z);

diagon = diag(z > 0 );
delta = diagon*W*d;



end

