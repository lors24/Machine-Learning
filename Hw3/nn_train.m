function [F,I,results,Xt,classt] = nn_train(name,sizes,tau,kappa,max_epoch,alpha)

[X,Y,class,Xv,Yv,classv,Xt,Yt,classt] = ex3(name);

[m, ~] = size(X); %number of training samples

%F = zeros(max_epoch/alpha,2); %stores results. First
% column for objective function and second column for accuracy.

%Create network
[L, weights, biases] = Network(sizes);
c = 1;
for e=1:max_epoch
    order = randperm(m);
    Xrnd = X(order,:);
    Yrnd = Y(order,:);
    %classrnd = class(order)+1;

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
    if mod(e-1,alpha)==0
        [error,accuracy] = eval_NN(Xv,Yv,classv,L,weights,biases); 
        F(c,1) = e;
        F(c,2) = error;
        F(c,3) = accuracy;
        
        fprintf('Epoch# %d \n', e);
        fprintf('Error %d \n',F(c,2));
        fprintf('Accuracy %d \n',F(c,3));
        
        if e>5*alpha && F(c,2)<=F(c-5,2)

            break
        end
        c = c+1;

    end
end
[train_error,train_accuracy] = eval_NN(X,Y,class,L,weights,biases); 
[test_error,test_accuracy,I] = eval_NN(Xt,Yt,classt,L,weights,biases); 

results = [train_error,train_accuracy; test_error,test_accuracy]
end

