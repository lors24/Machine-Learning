ntr = 200;
nv  = 150;
nts = 150;


train = [];
val = [];
test = [];
ntrain = [];
nval = [];
ntest =[];

index = eye(10);

for i=1:10
    [train_aux, val_aux, test_aux, ntrain_aux, nval_aux, ntest_aux]  = read_digits(i-1,ntr,nv,nts);
    train = [train; train_aux ones(ntr,1)*i-1 ones(ntr,10).*index(i,:)];
    val =[val; val_aux ones(nv,1)*i-1 ones(nv,10).*index(i,:)];
    test = [test; test_aux ones(nts,1)*i-1 ones(nts,10).*index(i,:)];
    ntrain = [ntrain; ntrain_aux ones(ntr,1)*i-1 ones(ntr,10).*index(i,:)];
    nval = [nval; nval_aux ones(nv,1)*i-1 ones(nv,10).*index(i,:)];
    ntest = [ntest; ntest_aux ones(nts,1)*i-1 ones(nts,10).*index(i,:)];
end



