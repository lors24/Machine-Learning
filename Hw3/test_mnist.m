
%digits;

X = ntrain(:,1:784);
Y = ntrain(:,786:end);

% 
Xv = nval(:,1:784);
Yv = nval(:,786:end);
class = nval(:,785);

F = nn_train(X,Y,Xv,Yv,class,[784 1000 10],100000,0.51,10);
