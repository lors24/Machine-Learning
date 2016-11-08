
name = '4';
train= importdata(strcat('data/data',name,'_train.csv'));
X = train(:,1:2);
class = (train(:,3)+1)/2;
%encode
Y = [class==0 class==1];

F = nn_train(X,Y,class,[2 100 2],tau,kappa,10);

% 
% validate = importdata(strcat('data/data',name,'_validate.csv'));
% Xv = validate(:,1:2);
% classv = validate(:,3);
% Yv = [classv == -1 classv==1];