function [ypred, ppred] = kernLRTest(c, Xtr, kernel, kerpar, Xts)
    K = KernelMatrix(Xts, Xtr, kernel, kerpar);
    ypred = K*c;
    ppred = exp(ypred)./(1+exp(ypred));
end