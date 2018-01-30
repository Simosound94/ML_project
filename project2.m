clear;
clc;
close all;

%Done in python
% - Stemming
% - Lemmatization
% - Stopwords removal
% - Bag of Words

load('X.mat')
load('Y.mat')
load('dictionary.mat')

%% Count number of words
count = sum(X,2);
X = [X, count];
[n,d] = size(X);
dictionary{end+1} = 'num_words';



%% Normalization

for j = 1:d
    me = mean(X(:,j));
    X(:,j) = (X(:,j)-me);
end


%% Seleziono un sottoinsieme
Xfull = X;
Yfull = Y;
i =randperm(n);
X = Xfull(i(1:1000),:);
Y = Yfull(i(1:1000),:);
[n,d] = size(X);


%% Kernel Regularized Logistic Regression with OVO
c = 3;
nLambda = 10;
nK=10;
kernel='gaussian';
%sigma = 9.5; %SIGMA MEAN NORM
sigma = 2;
% derivata da: mean(mean(pdist2(X,X))) scelgo sigma by default come la distanza media tra i punti
errToStop = 1e-8;
iter = 100000;

[~,YP] = max(Y, [],2);
errTrain = [];
errTest = [];
%prendo ugual numero di tutte le classi
idx = find(YP ==1);
i1 = idx(randperm(length(idx)))';
idx = find(YP ==2);
i2 = idx(randperm(length(idx)))';
idx = find(YP ==3);
i3 = idx(randperm(length(idx)))';
x = floor(length(i1)/nK);
i1 = reshape(i1(:,1:x*nK), [nK,x]);
x = floor(length(i2)/nK);
i2 = reshape(i2(:,1:x*nK), [nK,x]);
x = floor(length(i3)/nK);
i3 = reshape(i3(:,1:x*nK), [nK,x]);


lambdas = logspace(-6,2,nLambda);
for lambda = lambdas
    %fprintf('\n%f\n',lambda);
    errTr = 0;
    errTe= 0;
    for k =1:nK
        %  fprintf('%d\t',k);
        iv = [i1(k,:) i2(k,:), i3(k,:)]';
        nv = length(iv);
        il=[];
        for i = 1:nK
            if(i ~=k)
                il= [il i1(i,:) i2(i,:) i3(i,:)];
            end
        end
        il = il';
        nl = length(il);

        W=cell(c*(c-1)/2,1); 
        INDEX = cell(c*(c-1)/2,1);
        im = 0;
        for i = 1:c
            for j=i+1:c
                im = im+1;
                fm = YP(il) ==i;  %classe i ("negativa") solo quelle del LEARNING
                fp = YP(il) == j;    %classe j ("positiva") solo quelle del LEARNING
                ilp = [il(fm); il(fp)]; %per questo problema prendo solo gli indici
                % del training set (il) per cui i label siano i o j
                YPP = [-ones(sum(fm),1); ones(sum(fp),1)];
                
                w = kernLRTrain(X(ilp,:), YPP, kernel, sigma, lambda, iter, errToStop);
                W{im}= w; %è un concatenamento con i cell
                %non ricalcolo mai la stessa quantità, utilizzo solamente
                %diverse porzioni della stessa matrice secondo quello che
                %mi serve
                INDEX{im}= ilp;
            end
        end
        %    fprintf('tr\t')
        %Test su train e validation
        %TRAINING
        im = 0;
        YPred = [];
        for i = 1:c
            for j = i+1:c
                im = im + 1;
                %classifico tra -1,1 in tmp, ma poi lo devo ritrasformare se classe
                %i o j
                tmp = kernLRTest(W{im}, X(INDEX{im},:), kernel, sigma, X(il,:));
                tmp(tmp>0) = j;
                tmp(tmp<=0) = i;
                YPred = [YPred, tmp]; %#ok<AGROW>
            end
        end
        YPred1 = mode(YPred,2);
        errTr = errTr + mean(YPred1 ~=YP(il));

        %TEST
        im = 0;
        YPred = [];
        for i = 1:c
            for j = i+1:c
                im = im + 1;
                %classifico tra -1,1 in tmp, ma poi lo devo ritrasformare se classe
                %i o j
                tmp = kernLRTest(W{im}, X(INDEX{im},:), kernel, sigma, X(iv,:));
                tmp(tmp>0) = j;
                tmp(tmp<=0) = i;
                YPred = [YPred, tmp]; %#ok<AGROW>
            end
        end
        YPred1 = mode(YPred,2);
        errTe = errTe + mean(YPred1 ~=YP(iv));
    end
    errTrain = [errTrain, errTr/nK];
    errTest = [errTest errTe/nK];
    fprintf('lambda = %f Train error= %f Test error= %f\n',lambda, errTr/nK, errTe/nK);
end


 

%% Result for thw hole dataset but <1000 iter

% lambda = 0.000100 Train error= 0.425722 Test error= 0.450013
% lambda = 0.000774 Train error= 0.571443 Test error= 0.573997
% lambda = 0.005995 Train error= 0.575977 Test error= 0.577420
% lambda = 0.046416 Train error= 0.595594 Test error= 0.595862
% lambda = 0.359381 Train error= 0.595824 Test error= 0.596066
% lambda = 2.782559 Train error= 0.595824 Test error= 0.596066
% lambda = 21.544347 Train error= 0.595824 Test error= 0.596066
% lambda = 166.810054 Train error= 0.595824 Test error= 0.596066
% lambda = 1291.549665 Train error= 0.595824 Test error= 0.596066
% lambda = 10000.000000 Train error= 0.595824 Test error= 0.596066


trError = [0.425722 0.571443 0.575977 0.595594 0.595824 0.595824 0.595824 0.595824 0.595824 0.595824];
teError = [0.450013 0.573997 0.577420 0.595862 0.596066 0.596066 0.596066 0.596066 0.596066 0.596066];
figure
semilogx(lambdas, trError, 'b');
hold on
semilogx(lambdas, teError, 'r');
xlabel('\lambda') % x-axis label
ylabel('Median error') % y-axis label
legend('Training error', 'Validation error');



%% Only 1000 samples but <10000 iter

% lambda = 0.000001 Train error= 0.039342 Test error= 0.376531
% lambda = 0.000008 Train error= 0.123469 Test error= 0.415306
% lambda = 0.000060 Train error= 0.189229 Test error= 0.463265
% lambda = 0.000464 Train error= 0.335601 Test error= 0.511224
% lambda = 0.003594 Train error= 0.537188 Test error= 0.578571
% lambda = 0.027826 Train error= 0.584694 Test error= 0.598980
% lambda = 0.215443 Train error= 0.595465 Test error= 0.613265
% lambda = 1.668101 Train error= 0.597846 Test error= 0.602041
% lambda = 12.915497 Train error= 0.598866 Test error= 0.602041
% lambda = 100.000000 Train error= 0.598866 Test error= 0.602041
% 
trError = [0.039342 0.123469 0.189229 0.335601 0.537188 0.584694 0.595465 0.597846 0.598866 0.598866];
teError = [0.376531 0.415306 0.463265 0.511224 0.578571 0.598980 0.613265 0.602041 0.602041 0.602041];
figure
semilogx(lambdas, trError, 'b');
hold on
semilogx(lambdas, teError, 'r');
xlabel('\lambda') % x-axis label
ylabel('Median error') % y-axis label
legend('Training error', 'Validation error');

