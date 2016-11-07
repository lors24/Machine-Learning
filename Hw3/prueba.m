data = importdata('data_3class.csv');
[m n] = size(data);
data_encode = [data data(:,3)==0 data(:,3)==1 data(:,3)==2];

sizes = [2 4 3 3];
[layers, weights, biases] = Network(sizes);
a = [1;1]; %input

%training with feedforward
for i=1:layers-2
    a = feedforward(a,weights{i},biases{i},'ReLU');
end

y = feedforward(a,weights{end},biases{end},'Softmax')

