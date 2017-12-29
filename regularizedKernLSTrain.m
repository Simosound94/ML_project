function c = regularizedKernLSTrain(Ytr, Ktrain, lambda)
    n = length(Ytr);
    %K = KernelMatrix(Xtr, Xtr, kernel, sigma);
    c = (Ktrain + lambda*eye(n))\Ytr;
end
