data = importdata('data_3class.csv');
[m, n] = size(data);
data_encode = [data data(:,3)==0 data(:,3)==1 data(:,3)==2];

X = data_encode(:,1:2);
Y = data_encode(:,4:6);
class = data_encode(:,3);
sizes = [2 100 50 3];
tau = 100000;
kappa = 0.51;
max_epoch = 100;

F = train(X,Y,class,sizes,tau,kappa,max_epoch);
%F = train(X,Y,class,[2 200 3],1000000,0.9,100);
%F = train(X,Y,class,[2 200 3],100000,0.51,1000);

% 
% %eta = 0.00002;
% 
% 
% F = zeros(max_epoch,2);
% 
% 
% 
% [L, weights, biases] = Network(sizes);
% 
% for e=1:max_epoch
%     order = randperm(m);
%     data_rndm = data_encode(order,:);
% 
%     for k=1:m
%         a = cell(L,1);
%         delta = cell(L-1,1);
%         z = cell(L-1,1);
%         gW = cell(L-1,1); %weight gradients
%         gb = cell(L-1,1); %bias gradients
%         
%         a{1} = data_rndm(k,1:2)';
%         target = data_rndm(k,4:6)';
% 
%         %training with feedforward
%         for i=1:L-2
%              [a{i+1},z{i}] = feedforward(a{i},weights{i},biases{i},'ReLU');
%         end
% 
%         %Output layer
%         [a{end},z{end}] = feedforward(a{end-1},weights{end},biases{end},'Softmax');
% 
%         %Print error    
%         delta{end} = a{end}-target;
% 
%         for i=L-2:-1:1
%             delta{i} = backprop(delta{i+1},z{i},weights{i+1});
%         end
% 
%         %Gradient
%         
%         eta = (tau+(800*(e-1)+k))^-kappa;
% 
%         for i=1:L-1
%              gW{i} = a{i}*delta{i}';
%              gb{i} = delta{i};
%              weights{i} = weights{i}-eta*gW{i};
%              biases{i} = biases{i}-eta*gb{i};
%         end
%     end
%     
%     %Feedforward on all data
%     a = cell(L,1);
%     z = cell(L-1,1);
%     
%     a{1} = data_rndm(:,1:2)';
%     target = data_rndm(:,4:6)';
%     class = data_rndm(:,3)+1;
%     
%     for i=1:L-2
%          [a{i+1},z{i}] = feedforward(a{i},weights{i},biases{i},'ReLU');
%     end
% 
%     %Output layer
%     [a{end},z{end}] = feedforward(a{end-1},weights{end},biases{end},'Softmax');
%     F(e,1) = crossEntropy(a{end},target);
%     [~,I] = max(a{end});
%     
%     %Accuracy
%     F(e,2) = sum((I' == class))/m*100;
%     
% 
% end
% 
% plot(1:max_epoch,F)