function c = kernLRTrain(Xtr, Ytr, kernel, kerpar, lambda, iter, epsilon)
    [n, d] = size(Xtr);
    c = zeros(n,1);
    K = KernelMatrix(Xtr, Xtr, kernel, kerpar);
    L = eigs(K, 1)/n + lambda;
    gamma = 1/L;
    j = 0;
    fold = 0;
    f = inf;
    while(j < iter && abs(f - fold) >= epsilon)
        fold = f;
        j = j + 1;
        p = exp(Ytr.*(K*c));
        c = c - gamma*(-(1/n)*(Ytr./(1+p)) + 2*lambda*c);
        f = sum(log(1 + exp(-Ytr.*(K*c))))/n + lambda*c'*c;
%         if(mod(j,100)==0)
%            tmp = sign(kernLRTest(c, K));
%            tmp = sum(Ytr ~=tmp)/n;
%            disp(['iter:', num2str(j),'  err:', num2str(tmp)]);
%         end
    end
%     tmp = sign(kernLRTest(c, K));
%     tmp = sum(Ytr ~=tmp)/n;
%     disp(['iter:', num2str(j),'  err:', num2str(tmp)]);
end