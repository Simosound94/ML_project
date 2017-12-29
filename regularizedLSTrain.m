function w = regularizedLSTrain(Xtr, Ytr, lambda)
	n = size(Xtr,1);
    d = size(Xtr,2);
    % complete here, check the backslash command
    A = Xtr'*Xtr+lambda*diag(ones(d,1));
    b = Xtr'*Ytr;
    w=A\b;
end