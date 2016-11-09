function [train, val, test, train_norm, val_norm, test_norm ] = read_digits(n,ntr,nv,ntst)
%Read de mnist data base for digit 'n'
%Returns training, validation and test set comprised of the following rows:
%   training: 1-200
%   validation:201-350
%   test: 351-500
%Also returns the normalized versions of each data set. Normalization is:
% 2X/255-1 (to map all the values between -1 and 1)

s = num2str(n);
data = importdata(strcat('data/mnist_digit_',s,'.csv'));
train = data(1:ntr,:);
val = data(ntr+1:ntr+nv,:);
test = data(ntr+nv+1:ntr+nv+ntst,:);
train_norm = train*2/255-1;
val_norm = val*2/255-1;
test_norm = test*2/255-1;
end

