[X,Y,class,Xv,Yv,classv,Xt,Yt,classt] = ex3('4');

%Dataset 1:


%F1 = nn_train(X,Y,Xv,Yv,Xt,Yt,class,classv,classt,[2 100 2],100000,0.5,10,1);
F2 = nn_train(X,Y,Xv,Yv,classv,[2 5 2],100000,0.8,150,10); %more time
F3 = nn_train(X,Y,Xv,Yv,classv,[2 10 10 2],100000,0.8,100,1); %more time
%F4 = nn_train(X,Y,Xv,Yv,classv,[2 50 50 2],100000,0.8,10,1);
F5 = nn_train(X,Y,Xv,Yv,classv,[2 5 5 2],100000,0.8,100,1);

sizes = [2 100 2];
tau = 100000;
kappa = 0.8;
max_epoch = 1000;
alpha = 10;

[F,I,results,Xt,class] = nn_train('1',sizes,tau,kappa,max_epoch,alpha);
plotNN('1',Xt,class,I)




%F = nn_train(X,Y,Xv,Yv,class,[2 100 2],100000,0.8,1000,10); 1,2: 100, 88


%4:

 %F1 = nn_train(X,Y,Xv,Yv,Xt,Yt,class,classv,classt,[2 100 2],1000000,0.8,1000,1)
