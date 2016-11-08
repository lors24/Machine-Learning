function [F] = nn_train(X,Y,Xv,Yv,class,sizes,tau,kappa,max_epoch)

[m, ~] = size(X); %number of training samples

F = zeros(max_epoch,2); %stores results. First
% column for objective function and second column for accuracy.

%Create network
[L, weights, biases] = Network(sizes);

for e=1:max_epoch
    order = randperm(m);
    Xrnd = X(order,:);
    Yrnd = Y(order,:);
    classrnd = class(order)+1;

    for k=1:m %for every observation in the training sample
        a = cell(L,1); %activations
        
        z = cell(L-1,1); %inputs
        delta = cell(L-1,1); %deltas for backprop
        
        gW = cell(L-1,1); %weight gradients
        gb = cell(L-1,1); %bias gradients
        
        %First activation is the input
        a{1} = Xrnd(k,:)';
        target = Yrnd(k,:)';

        %training with feedforward
        for i=1:L-2
             [a{i+1},z{i}] = feedforward(a{i},weights{i},biases{i},'ReLU');
        end

        %Output layer
        [a{end},z{end}] = feedforward(a{end-1},weights{end},biases{end},'Softmax');

        %Backprop
        delta{end} = a{end}-target; %L

        for i=L-2:-1:1
            delta{i} = backprop(delta{i+1},z{i},weights{i+1});
        end

        %Gradient
        
        eta = (tau+(800*(e-1)+k))^-kappa;

        for i=1:L-1
             gW{i} = a{i}*delta{i}';
             gb{i} = delta{i};
             weights{i} = weights{i}-eta*gW{i};
             biases{i} = biases{i}-eta*gb{i};
        end
    end
    
    %Feedforward on all data
    a = cell(L,1);
    z = cell(L-1,1);
    
    a{1} = Xv';
    target = Yv';
    
    for i=1:L-2
         [a{i+1},z{i}] = feedforward(a{i},weights{i},biases{i},'ReLU');
    end

    %Output layer
    [a{end},z{end}] = feedforward(a{end-1},weights{end},biases{end},'Softmax');
    
    F(e,1) = crossEntropy(a{end},target);
    [~,I] = max(a{end});
    
    %Accuracy
    F(e,2) = sum((I' == classrnd))/m*100;
    

end

end

