
name = '2';
train= importdata(strcat('data/data',name,'_train.csv'));
X = train(:,1:2);
class = (train(:,3)+1)/2;
%encode
Y = [class==0 class==1];


% 
validate = importdata(strcat('data/data',name,'_validate.csv'));
Xv = validate(:,1:2);
class2 = (validate(:,3)+1)/2;
Yv = [class2 == 0 class2==1];

F = nn_train(X,Y,Xv,Yv,class2,[2 100 2],100000,0.51,1000);
