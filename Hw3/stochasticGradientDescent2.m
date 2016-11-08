function [R,opt] = stochasticGradientDescent2(M,W0,tau,kappa,epsilon)

data = importdata('curvefittingp2.txt');

AX = data(1,:)'; 
Y = data(2,:)';

W=W0;
gradient=zeros(M+1,1);

for l=0:M
    X(:,l+1)=AX.^l;
end

s=size(Y);
opt=polynomialRegression(M);
Z(1,:)=W0';
i=2;
n=norm(W0-opt);
Z(1,M+2)=norm(gradient);
Z(1,M+3)=n;
Z(1,M+4)=1;

while(n>epsilon && i<=100000)
    for k=1:s(1)
        eta=(tau+i-1)^(-kappa);
        gradient=2*(X(k,:)*W-Y(k))*X(k,:)'; 
        W=W-eta*gradient;
        n=norm(W-opt);
        Z(i,:)=[W',norm(gradient),n,i];
        i=i+1;  
    end
     %random = randperm(s(1));
     %X = X(random');
     %Y = Y(random');
 
end 
    figure
    subplot(2,1,1);
    plot(Z(:,M+4),Z(:,M+2));
    title('Iteration vs. gradient norm');
    xlabel('i');
    ylabel('Gradient norm');
    subplot(2,1,2);
    plot(Z(:,M+4),Z(:,M+3));
    title('Iteration vs. Error norm');
    xlabel('i');
    ylabel('Norm(error)');
    R=Z(i-1,:);

end

% [S,opt]=stochasticGradientDescent2(2,ones(3,1),0,0.501,1.1984)
% [S,opt]=stochasticGradientDescent2(3,zeros(4,1),0,0.501,6.007)