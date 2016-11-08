function [out] = Softmax(z)

out = exp(z)./sum(exp(z));

end

