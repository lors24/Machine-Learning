function [ output_args ] = train(X,Y,sizes, tau,kappa,max_epoch)

% [m, ~] = size(data); F = zeros(max_epoch,2); %stores results. First
% column for objective function and second column for accuracy.

%Create network
[L, weights, biases] = Network(sizes);

for e=1:max_epoch
    order = randperm(m);
    Xrnd = X(order,:);
    Yrnd = Y(order,:);

    for k=1:m %for every observation in the training sample
        a = cell(L,1); %activations
        
        z = cell(L-1,1); %inputs
        delta = cell(L-1,1); %deltas for backprop
        
        gW = cell(L-1,1); %weight gradients
        gb = cell(L-1,1); %bias gradients
        
        %First activation is the input
        a{1} = data_rndm(k,1:2)';
        target = data_rndm(k,4:6)';

        %training with feedforward
        for i=1:L-2
             [a{i+1},z{i}] = feedforward(a{i},weights{i},biases{i},'ReLU');
        end

        %Output layer
        [a{end},z{end}] = feedforward(a{end-1},weights{end},biases{end},'Softmax');

        %Print error
        delta{end} = a{end}-target;

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
    
    a{1} = data_rndm(:,1:2)';
    target = data_rndm(:,4:6)';
    class = data_rndm(:,3)+1;
    
    for i=1:L-2
         [a{i+1},z{i}] = feedforward(a{i},weights{i},biases{i},'ReLU');
    end

    %Output layer
    [a{end},z{end}] = feedforward(a{end-1},weights{end},biases{end},'Softmax');
    F(e,1) = crossEntropy(a{end},target);
    [~,I] = max(a{end});
    
    %Accuracy
    F(e,2) = sum((I' == class))/m*100;
    

end

plot(1:max_epoch,F)
end

