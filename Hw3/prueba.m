data = importdata('data_3class.csv');
[m, n] = size(data);
data_encode = [data data(:,3)==0 data(:,3)==1 data(:,3)==2];

sizes = [2 4 5 2 3];
[L, weights, biases] = Network(sizes);


a = cell(L,1);
delta = cell(L-1,1);
z = cell(L-1,1);

a{1} = [1;1]; %input of the network


%training with feedforward
for i=1:L-2
    [a{i+1},z{i}] = feedforward(a{i},weights{i},biases{i},'ReLU');
end

%Output layer
[a{end},z{end}] = feedforward(a{end-1},weights{end},biases{end},'Softmax');
target = data_encode(1,4:6)';

delta{end} = a{end}-target;

for i=L-2:-1:1
    delta{i} = backprop(delta{i+1},z{i},weights{i+1});
end

%Gradient

gW = cell(L-1,1); %weight gradients
gb = cell(L-1,1); %bias gradients

for i=1:L-1
    gW{i} = a{i}*delta{i}';
    gb{i} = delta{i};
end




