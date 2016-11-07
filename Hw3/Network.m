function [layers, weights, biases] = Network( sizes )
%Initializes the network given the sizes of each layer
%The weights are initialized as samples of a normal gaussian with mean 0
%and standard deviation 1/sqrt(m). 

layers = length(sizes);
weights = cell(layers-1,1);
biases = cell(layers-1,1);

for i=1:layers-1
    m = sizes(i);
    n = sizes(i+1);
    weights{i} = normrnd(0,1/sqrt(m),m,n);
    biases{i} = zeros(n,1);
end



end

