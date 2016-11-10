function [X,Y,class,Xv,Yv,classv,Xt,Yt,classt] = ex3(name)

train= importdata(strcat('data/data',name,'_train.csv'));
X = train(:,1:2);
class = (train(:,3)+1)/2;
%encode
Y = [class==0 class==1];


% 
validate = importdata(strcat('data/data',name,'_validate.csv'));
Xv = validate(:,1:2);
classv = (validate(:,3)+1)/2;
Yv = [classv == 0 classv==1];

test= importdata(strcat('data/data',name,'_train.csv'));
Xt = test(:,1:2);
classt = (test(:,3)+1)/2;
%encode
Yt = [classt==0 classt==1];


end

