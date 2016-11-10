function [error,accuracy,I] = eval_NN(X,Y,class,L,weights,biases)
[m ~]=size(X);

 a = cell(L,1);
 z = cell(L-1,1);
    
a{1} = X';
target = Y';
    
for i=1:L-2
     [a{i+1},z{i}] = feedforward(a{i},weights{i},biases{i},'ReLU');
end

%Output layer
[a{end},~] = feedforward(a{end-1},weights{end},biases{end},'Softmax');
    
%Error   
error = crossEntropy(a{end},target);
[~,I] = max(a{end});
    
%Accuracy
class_eval = class+1;
accuracy = sum((I' == class_eval))/m*100;
        
end

