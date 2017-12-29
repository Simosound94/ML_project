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

%% Popular words 

word_count = sum(X);
[count,I]=sort(word_count,'descend');
fprintf('%s\t', dictionary{I(1:40)});
fprintf('\n');
fprintf('%d\t', count(1:40));
fprintf('\n');
% time	man	day	thing	eye	said	did	old	like	life	
% night	thought	little	great	long	saw	say	house	came	
% hand	year	word	death	heart	mind	place	friend	
% far	know	shall	heard	men	light	left	door	felt
% room	love	come	earth	


%% Count number of words
count = sum(X,2);
X = [X, count];
[n,d] = size(X);
dictionary{end+1} = 'num_words';



%% Normalization

% ONLY MEAN NORMALIZED DATA
for j = 1:d
    me = mean(X(:,j));
    X(:,j) = (X(:,j)-me);
end

%% High dimension distances

i = randperm(n);
size_sample = 4000;

distances = pdist2(X(i(1:size_sample),:), X(i(1:size_sample),:));
fprintf('distances compleated \n')
ma = max(max(distances));
num = [];
for dist = linspace(0, ma, 30);
    num = [num sum(sum(distances<=dist))];
end
num = num ./(size_sample^2);

figure; box on; grid on;
plot(linspace(0, ma, 30), num);
xlabel('distance')
ylabel('% points')



%% Compute approximation cost of PCA
% Note: Data mean normalized

%Per vedere direzioni di massima varianza dobbiamo usare PCA su dati solo
%mean normalized, se li facciamo anche variance normalized tutte le
%features risuteranno appartenere a [0, 1] ed il risultato non darà le
%parole più comuni


%[V, d] = PCA(X, 13000); %per velocità ho già salvato il risultato
load('Vd_only_mean_norm_PCA');


% features = [3 10 100 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000];
% c=[];
% for k =features
%    cost =  sum(sum((X - X*V(:,1:k)*V(:,1:k)').^2,2))/n;
%    c = [c cost];
%    fprintf('k = %d \t cost= %f\n',k,cost);    
% end
% plot(features, c);

% 
% k = 3 	 cost= 11.477598
% k = 10 	 cost= 11.194886
% k = 100 	 cost= 9.477404
% k = 1000 	 cost= 4.907257
% k = 2000 	 cost= 3.020597
% k = 3000 	 cost= 1.973920
% k = 4000 	 cost= 1.319777
% k = 5000 	 cost= 0.885796
% k = 6000 	 cost= 0.589035
% k = 7000 	 cost= 0.383497
% k = 8000 	 cost= 0.240759
% k = 9000 	 cost= 0.142402
% k = 10000 	 cost= 0.076502
% k = 11000 	 cost= 0.034919
% k = 12000 	 cost= 0.011782
% k = 13000 	 cost= 0.001778


%% PCA Results

k = [3 10 100 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000];
cost = [11.477598 11.194886 9.477404 4.907257 3.020597 1.973920 1.319777 0.885796 0.589035 0.383497 0.240759 0.142402 0.076502 0.034919 0.011782 0.001778];

plot(k,cost);
title('cost');
xlabel('num of features');



% la prima direzione V(:,1) è quella dove vi è più varianza
% i componenti maggiori di questa direzione sono quelli ove i dati vengono
% proiettati di più
[~, I] = sort(V(:,1),'descend'); %indici elementi più importanti
fprintf('max components: %s\n',sprintf('%s, ', dictionary{I(1:10)}));

var = sum((X - ones(n,1)*mean(X)).^2)/(n-1); %stimatore varianza delle varie features
[mdi, Idi] = sort(var,'descend');
fprintf('massima varianza: %s\n',sprintf('%s, ', dictionary{Idi(1:10)}));

%max components: num_words, eye, man, like, great, life, mind, earth, world, word, 
%massima varianza: num_words,eye,man,like, time, day,  thing, said, did,  life, 

%% Visualizzazione dati proiettati in 3 dimensioni
X_proj = X*V(:,1:3);

Ytr(Y(:,2)==1) = 1;
Ytr(Y(:,2)==1) = 2;
Ytr(Y(:,3)==1) = 3;
figure;
scatter3(X_proj(:,1),X_proj(:,2),X_proj(:,3),25,Ytr);
figure;

scatter(X_proj(:,1),X_proj(:,2),25,Ytr);
[~, I] = sort(V(:,1),'descend');
xlabel([dictionary{I(1)},', ',dictionary{I(2)},', ',dictionary{I(3)}]);
[~, I] = sort(V(:,2),'descend');
ylabel([dictionary{I(1)},', ',dictionary{I(2)},', ',dictionary{I(3)}]);
xlim([0 40]);
ylim([-2 4]);


%% Linear regression with OVO

c = 3;
nLambda = 15;
nK=5;


[~,YP] = max(Y, [],2);
errTrain = [];
errTest = [];
%prendo ugual numero di tutte le classi per fare "stratified sampling"
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


for lambda = logspace(-4,4,nLambda)
    fprintf('\n%f\n',lambda);
    errTr = 0;
    errTe= 0;
    for k =1:nK
        fprintf('%d\t',k);
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

        W=[];
        B =[];
        im = 0;
        for i = 1:c
            for j=i+1:c
                im = im+1;
                fm = YP(il) ==i;  %classe i ("negativa") solo quelle del LEARNING
                fp = YP(il) == j;    %classe j ("positiva") solo quelle del LEARNING
                ilp = [il(fm); il(fp)]; %per questo problema prendo solo gli indici
                % del training set (il) per cui i label siano i o j
                YPP = [-ones(sum(fm),1); ones(sum(fp),1)];
                meanY = mean(YPP);
                YPP = YPP - meanY*ones(sum(fm)+sum(fp),1);
                w = regularizedLSTrain(X(ilp, :) , YPP, lambda);
                W =[W w];
                B =[B meanY];
            end
        end

        %Test su train e validation
        %TRAINING
        im = 0;
        YPred = [];
        for i = 1:c
            for j = i+1:c
                im = im + 1;
                %classifico tra -1,1 in tmp, ma poi lo devo ritrasformare se classe
                %i o j
                tmp = X(il,:)*W(:,im)+B(im);
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
                tmp = X(iv,:)*W(:,im)+B(im);
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
%error computed as  mean of miss classification

% lambda = 0.000100 Train error= 0.003716 Test error= 0.578851
% lambda = 0.000373 Train error= 0.004227 Test error= 0.558212
% lambda = 0.001389 Train error= 0.005326 Test error= 0.527561
% lambda = 0.005179 Train error= 0.006756 Test error= 0.491699
% lambda = 0.019307 Train error= 0.008748 Test error= 0.448276
% lambda = 0.071969 Train error= 0.012069 Test error= 0.394943
% lambda = 0.268270 Train error= 0.017344 Test error= 0.337471
% lambda = 1.000000 Train error= 0.026117 Test error= 0.280817
% lambda = 3.727594 Train error= 0.045709 Test error= 0.239336
% lambda = 13.894955 Train error= 0.086590 Test error= 0.225492
% lambda = 51.794747 Train error= 0.155773 Test error= 0.243576
% lambda = 193.069773 Train error= 0.253040 Test error= 0.303040
% lambda = 719.685673 Train error= 0.384163 Test error= 0.407714
% lambda = 2682.695795 Train error= 0.524598 Test error= 0.533640
% lambda = 10000.000000 Train error= 0.587791 Test error= 0.589783


lambda = [0.000100, 0.000373, 0.001389, 0.005179, 0.019307, 0.071969, 0.268270, 1.000000, 3.727594, 13.894955, 51.794747, 193.069773, 719.685673, 2682.695795 10000.000000];
trError = [0.003716 0.004227 0.005326 0.006756 0.008748 0.012069 0.017344 0.026117 0.045709 0.086590 0.155773 0.253040 0.384163 0.524598 0.587791];
teError = [0.578851 0.558212 0.527561 0.491699 0.448276 0.394943 0.337471 0.280817 0.239336 0.225492 0.243576 0.303040 0.407714 0.533640 0.589783];


figure
semilogx(lambda, trError, 'b');
hold on
semilogx(lambda, teError, 'r');
xlabel('\lambda') % x-axis label
ylabel('Median error') % y-axis label
legend('Validation error','Training error');




%% Kernel Logistic Regression with OVO
c = 3;
nLambda = 15;
nK=5;
kernel='gaussian';
sigma = 9.5; %SIGMA MEAN NORM
% sigma = 7;
% derivata da: mean(mean(pdist2(X,X))) scelgo sigma by default come la distanza media tra i punti


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

%una volta per tutte calcolo la kernel Matrix, di train e di test (a seconda dei punti che prendo di K)

%K = KernelMatrix(X, X, kernel, sigma); %per velocita ho gia salvato il
%risultato
load('K_mean_norm.mat')

for lambda = logspace(-4,4,nLambda)
    % fprintf('\n%f\n',lambda);
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
                
                w = regularizedKernLSTrain(YPP, K(ilp, ilp), lambda);
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
                tmp = regularizedKernLSTest(W{im}, K(il,INDEX{im}));
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
                tmp = regularizedKernLSTest(W{im}, K(iv,INDEX{im}));
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

%Having data also "range" normalized would help the gradient descent

% lambda = 0.000100 Train error= 0.001456 Test error= 0.308506
% lambda = 0.000373 Train error= 0.001622 Test error= 0.295479
% lambda = 0.001389 Train error= 0.002427 Test error= 0.277650
% lambda = 0.005179 Train error= 0.005172 Test error= 0.254100
% lambda = 0.019307 Train error= 0.014470 Test error= 0.225441
% lambda = 0.071969 Train error= 0.047152 Test error= 0.216347
% lambda = 0.268270 Train error= 0.110626 Test error= 0.236117
% lambda = 1.000000 Train error= 0.198557 Test error= 0.278876
% lambda = 3.727594 Train error= 0.298953 Test error= 0.348097
% lambda = 13.894955 Train error= 0.420894 Test error= 0.447050
% lambda = 51.794747 Train error= 0.553321 Test error= 0.558621
% lambda = 193.069773 Train error= 0.574585 Test error= 0.576909
% lambda = 719.685673 Train error= 0.583461 Test error= 0.583959
% lambda = 2682.695795 Train error= 0.593870 Test error= 0.594074


lambda = [0.000100, 0.000373, 0.001389, 0.005179, 0.019307, 0.071969, 0.268270, 1.000000, 3.727594, 13.894955, 51.794747, 193.069773, 719.685673, 2682.695795];
trError = [0.001456 0.001622  0.002427 0.005172 0.014470 0.047152 0.110626 0.198557 0.298953 0.420894 0.553321 0.574585 0.583461 0.593870];
teError = [0.308506 0.295479 0.277650 0.254100 0.225441 0.216347 0.236117 0.278876 0.348097 0.447050 0.558621 0.576909 0.583959 0.594074];
figure
semilogx(lambda, trError, 'b');
hold on
semilogx(lambda, teError, 'r');
xlabel('\lambda') % x-axis label
ylabel('Median error') % y-axis label
legend('Validation error','Training error');



%% Kernel Logistic Regression and PCA
c = 3;
nLambda = 10;
lambdaSpace = logspace(-3,3,nLambda);
nK=5;
kernel='gaussian';


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

%una volta per tutte calcolo la kernel Matrix, di train e di test (a seconda dei punti che prendo di K)

lambdasOpt = [];
sigmasOpt = [];
costOpt = [];
for num_f = [13000 8000 5000 3000 1000 100 3]
    X_proj = X*V(:,1:num_f);
    %calcolo sigma prendendo un po' di punti random
    i = randperm(n);
    sigma = mean(mean(pdist2(X_proj(i(1:2000),:),X_proj(i(1:2000),:))));
    sigmasOpt = [sigmasOpt, sigma];
    K = KernelMatrix(X_proj, X_proj, kernel, sigma);
    fprintf('\n --FEATURES: %d', num_f);
    errTrain = [];
    errTest = [];
    for lambda = lambdaSpace
        %fprintf('\n%f\n',lambda);
        errTr = 0;
        errTe= 0;
        for k =1:nK
            %fprintf('%d\t',k);
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

                    w = regularizedKernLSTrain(YPP, K(ilp, ilp), lambda);
                    W{im}= w; %è un concatenamento con i cell
                    %non ricalcolo mai la stessa quantità, utilizzo solamente
                    %diverse porzioni della stessa matrice secondo quello che
                    %mi serve
                    INDEX{im}= ilp;
                end
            end
            %Test su train e validation
            %TRAINING
            im = 0;
            YPred = [];
            for i = 1:c
                for j = i+1:c
                    im = im + 1;
                    %classifico tra -1,1 in tmp, ma poi lo devo ritrasformare se classe
                    %i o j
                    tmp = regularizedKernLSTest(W{im}, K(il,INDEX{im}));
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
                    tmp = regularizedKernLSTest(W{im}, K(iv,INDEX{im}));
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
    [cOpt, iOpt] = min(errTest);
    costOpt = [costOpt cOpt];
    lambdasOpt = [lambdasOpt, lambdaSpace(iOpt)];
end

% sigma =
% 
%     9.3382
% 
% 
%  --FEATURES: 13000
% lambda = 0.001000 Train error= 0.002082 Test error= 0.277497
% lambda = 0.004642 Train error= 0.004598 Test error= 0.249400
% lambda = 0.021544 Train error= 0.015300 Test error= 0.225032
% lambda = 0.100000 Train error= 0.058799 Test error= 0.220434
% lambda = 0.464159 Train error= 0.143895 Test error= 0.250473
% lambda = 2.154435 Train error= 0.253908 Test error= 0.316884
% lambda = 10.000000 Train error= 0.380945 Test error= 0.412261
% lambda = 46.415888 Train error= 0.545683 Test error= 0.552133
% lambda = 215.443469 Train error= 0.573346 Test error= 0.576194
% lambda = 1000.000000 Train error= 0.582516 Test error= 0.583244
% 
% sigma =
% 
%     9.4285
% 
% 
%  --FEATURES: 8000
% lambda = 0.001000 Train error= 0.002197 Test error= 0.288276
% lambda = 0.004642 Train error= 0.005287 Test error= 0.255581
% lambda = 0.021544 Train error= 0.020690 Test error= 0.226156
% lambda = 0.100000 Train error= 0.064738 Test error= 0.220639
% lambda = 0.464159 Train error= 0.148391 Test error= 0.250524
% lambda = 2.154435 Train error= 0.256450 Test error= 0.317548
% lambda = 10.000000 Train error= 0.382605 Test error= 0.413333
% lambda = 46.415888 Train error= 0.546769 Test error= 0.552848
% lambda = 215.443469 Train error= 0.572580 Test error= 0.575019
% lambda = 1000.000000 Train error= 0.582158 Test error= 0.583091
%
% sigma =
% 
%     9.7471
% 
%  --FEATURES: 5000
%  lambda = 0.001000 Train error= 0.003282 Test error= 0.303193
% lambda = 0.004642 Train error= 0.010166 Test error= 0.263346
% lambda = 0.021544 Train error= 0.033155 Test error= 0.230856
% lambda = 0.100000 Train error= 0.081047 Test error= 0.218544
% lambda = 0.464159 Train error= 0.159936 Test error= 0.251239
% lambda = 2.154435 Train error= 0.263704 Test error= 0.318314
% lambda = 10.000000 Train error= 0.388978 Test error= 0.417778
% lambda = 46.415888 Train error= 0.549693 Test error= 0.555811
% lambda = 215.443469 Train error= 0.572733 Test error= 0.575990
% lambda = 1000.000000 Train error= 0.587216 Test error= 0.588046
% 
% sigma =
% 
%     9.8151
% 
% 
%  --FEATURES: 3000
% lambda = 0.001000 Train error= 0.005951 Test error= 0.300434
% lambda = 0.004642 Train error= 0.021481 Test error= 0.265390
% lambda = 0.021544 Train error= 0.055364 Test error= 0.232184
% lambda = 0.100000 Train error= 0.102695 Test error= 0.223908
% lambda = 0.464159 Train error= 0.174636 Test error= 0.253742
% lambda = 2.154435 Train error= 0.270856 Test error= 0.316884
% lambda = 10.000000 Train error= 0.391149 Test error= 0.417420
% lambda = 46.415888 Train error= 0.550613 Test error= 0.555249
% lambda = 215.443469 Train error= 0.573257 Test error= 0.575377
% lambda = 1000.000000 Train error= 0.586513 Test error= 0.587024
%
% sigma =
% 
%     8.9196
%
% 
%  --FEATURES: 1000
% lambda = 0.001000 Train error= 0.032248 Test error= 0.290932
% lambda = 0.004642 Train error= 0.073167 Test error= 0.274636
% lambda = 0.021544 Train error= 0.120639 Test error= 0.263346
% lambda = 0.100000 Train error= 0.161660 Test error= 0.255939
% lambda = 0.464159 Train error= 0.209387 Test error= 0.265441
% lambda = 2.154435 Train error= 0.282822 Test error= 0.317650
% lambda = 10.000000 Train error= 0.386411 Test error= 0.404802
% lambda = 46.415888 Train error= 0.540792 Test error= 0.545441
% lambda = 215.443469 Train error= 0.579387 Test error= 0.580332
% lambda = 1000.000000 Train error= 0.583218 Test error= 0.583499
% 
% sigma =
% 
%     8.3378
% 
% 
%  --FEATURES: 100
% lambda = 0.001000 Train error= 0.227739 Test error= 0.344623
% lambda = 0.004642 Train error= 0.282746 Test error= 0.352439
% lambda = 0.021544 Train error= 0.319796 Test error= 0.357292
% lambda = 0.100000 Train error= 0.338161 Test error= 0.362708
% lambda = 0.464159 Train error= 0.351839 Test error= 0.368582
% lambda = 2.154435 Train error= 0.377203 Test error= 0.388710
% lambda = 10.000000 Train error= 0.428467 Test error= 0.435811
% lambda = 46.415888 Train error= 0.540894 Test error= 0.542631
% lambda = 215.443469 Train error= 0.580575 Test error= 0.581252
% lambda = 1000.000000 Train error= 0.583397 Test error= 0.583550
% 
% sigma =
% 
%     7.7626
% 
% 
%  --FEATURES: 3
% lambda = 0.001000 Train error= 0.525798 Test error= 0.529757
% lambda = 0.004642 Train error= 0.528608 Test error= 0.531596
% lambda = 0.021544 Train error= 0.533295 Test error= 0.536092
% lambda = 0.100000 Train error= 0.536041 Test error= 0.539566
% lambda = 0.464159 Train error= 0.537510 Test error= 0.540383
% lambda = 2.154435 Train error= 0.542976 Test error= 0.545236
% lambda = 10.000000 Train error= 0.563308 Test error= 0.563576
% lambda = 46.415888 Train error= 0.579693 Test error= 0.579464
% lambda = 215.443469 Train error= 0.583397 Test error= 0.583295
% lambda = 1000.000000 Train error= 0.583716 Test error= 0.584061

%%
close all;
num_f = [13000 8000 5000 3000 1000 100 3];
lambdasOpt = [0.1 0.1 0.1 0.1 0.1 0.001 0.001];
sigmasOpt = [9.338223 9.428485 9.7471 9.8151 8.919572 8.337811 7.762558];
costOpt = [0.220434 0.220639 0.218544 0.223908 0.255939 0.344623 0.529757];
figure;
plot(num_f, costOpt, 'b');
xlabel('number of features')
title('Cost opt');

figure;
semilogx(lambdasOpt, sigmasOpt);
xlabel('\lambda') % x-axis label
ylabel('sigma') % y-axis label



%% Regularized Least Squares and PCA

c = 3;
nLambda = 15;
nK=5;
lambdaSpace = logspace(-3,3,nLambda);

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

lambdasOpt = [];
costOpt = [];
for num_f = [13000 8000 5000 3000 1000 100 3]
    X_proj = X*V(:,1:num_f);
    fprintf('\n --FEATURES: %d', num_f);
     errTrain = [];
    errTest = [];
    for lambda = lambdaSpace
        %fprintf('\n%f\n',lambda);
        errTr = 0;
        errTe= 0;
        for k =1:nK
            %fprintf('%d\t',k);
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
            W=[];
            B =[];
            im = 0;
            for i = 1:c
                for j=i+1:c
                    im = im+1;
                    fm = YP(il) ==i;  %classe i ("negativa") solo quelle del LEARNING
                    fp = YP(il) == j;    %classe j ("positiva") solo quelle del LEARNING
                    ilp = [il(fm); il(fp)]; %per questo problema prendo solo gli indici
                    % del training set (il) per cui i label siano i o j
                    YPP = [-ones(sum(fm),1); ones(sum(fp),1)];
                    meanY = mean(YPP);
                    YPP = YPP - meanY*ones(sum(fm)+sum(fp),1);
                    w = regularizedLSTrain(X_proj(ilp, :) , YPP, lambda);
                    W =[W w];
                    B =[B meanY];
                end
            end

            %Test su train e validation
            %TRAINING
            im = 0;
            YPred = [];
            for i = 1:c
                for j = i+1:c
                    im = im + 1;
                    %classifico tra -1,1 in tmp, ma poi lo devo ritrasformare se classe
                    %i o j
                    tmp = X_proj(il,:)*W(:,im)+B(im);
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
                    tmp = X_proj(iv,:)*W(:,im)+B(im);
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
    [cOpt, iOpt] = min(errTest);
    costOpt = [costOpt cOpt];
    lambdasOpt = [lambdasOpt, lambdaSpace(iOpt)];
end



%  --FEATURES: 13000
% lambda = 0.001000 Train error= 0.005249 Test error= 0.542273
% lambda = 0.002683 Train error= 0.005977 Test error= 0.521328
% lambda = 0.007197 Train error= 0.007165 Test error= 0.491188
% lambda = 0.019307 Train error= 0.008825 Test error= 0.456705
% lambda = 0.051795 Train error= 0.011149 Test error= 0.414559
% lambda = 0.138950 Train error= 0.014432 Test error= 0.368582
% lambda = 0.372759 Train error= 0.019080 Test error= 0.318161
% lambda = 1.000000 Train error= 0.025619 Test error= 0.276833
% lambda = 2.682696 Train error= 0.038544 Test error= 0.244138
% lambda = 7.196857 Train error= 0.061941 Test error= 0.227893
% lambda = 19.306977 Train error= 0.101162 Test error= 0.226667
% lambda = 51.794747 Train error= 0.155913 Test error= 0.242095
% lambda = 138.949549 Train error= 0.225479 Test error= 0.282299
% lambda = 372.759372 Train error= 0.313550 Test error= 0.348404
% lambda = 1000.000000 Train error= 0.420562 Test error= 0.440562
% 
%  --FEATURES: 8000
% lambda = 0.001000 Train error= 0.025236 Test error= 0.451648
% lambda = 0.002683 Train error= 0.025453 Test error= 0.446437
% lambda = 0.007197 Train error= 0.026220 Test error= 0.433359
% lambda = 0.019307 Train error= 0.027139 Test error= 0.418186
% lambda = 0.051795 Train error= 0.028710 Test error= 0.391775
% lambda = 0.138950 Train error= 0.031034 Test error= 0.363525
% lambda = 0.372759 Train error= 0.035211 Test error= 0.326130
% lambda = 1.000000 Train error= 0.042452 Test error= 0.285773
% lambda = 2.682696 Train error= 0.053014 Test error= 0.250421
% lambda = 7.196857 Train error= 0.073218 Test error= 0.230192
% lambda = 19.306977 Train error= 0.108340 Test error= 0.225849
% lambda = 51.794747 Train error= 0.159221 Test error= 0.241635
% lambda = 138.949549 Train error= 0.227075 Test error= 0.281686
% lambda = 372.759372 Train error= 0.314189 Test error= 0.348301
% lambda = 1000.000000 Train error= 0.420792 Test error= 0.440307
% 
%  --FEATURES: 5000
% lambda = 0.001000 Train error= 0.073244 Test error= 0.323372
% lambda = 0.002683 Train error= 0.073423 Test error= 0.323065
% lambda = 0.007197 Train error= 0.073474 Test error= 0.321430
% lambda = 0.019307 Train error= 0.073487 Test error= 0.319591
% lambda = 0.051795 Train error= 0.073665 Test error= 0.316373
% lambda = 0.138950 Train error= 0.074138 Test error= 0.308097
% lambda = 0.372759 Train error= 0.075313 Test error= 0.294100
% lambda = 1.000000 Train error= 0.078225 Test error= 0.275709
% lambda = 2.682696 Train error= 0.084777 Test error= 0.250728
% lambda = 7.196857 Train error= 0.098250 Test error= 0.232899
% lambda = 19.306977 Train error= 0.124266 Test error= 0.227842
% lambda = 51.794747 Train error= 0.168161 Test error= 0.242350
% lambda = 138.949549 Train error= 0.231839 Test error= 0.281430
% lambda = 372.759372 Train error= 0.316335 Test error= 0.347842
% lambda = 1000.000000 Train error= 0.421520 Test error= 0.439847
% 
%  --FEATURES: 3000
% lambda = 0.001000 Train error= 0.126284 Test error= 0.277241
% lambda = 0.002683 Train error= 0.126232 Test error= 0.277395
% lambda = 0.007197 Train error= 0.126258 Test error= 0.277292
% lambda = 0.019307 Train error= 0.126156 Test error= 0.276322
% lambda = 0.051795 Train error= 0.125977 Test error= 0.275811
% lambda = 0.138950 Train error= 0.126169 Test error= 0.274432
% lambda = 0.372759 Train error= 0.126130 Test error= 0.271418
% lambda = 1.000000 Train error= 0.126335 Test error= 0.264623
% lambda = 2.682696 Train error= 0.128378 Test error= 0.253640
% lambda = 7.196857 Train error= 0.134023 Test error= 0.239489
% lambda = 19.306977 Train error= 0.150817 Test error= 0.233768
% lambda = 51.794747 Train error= 0.184227 Test error= 0.246130
% lambda = 138.949549 Train error= 0.240868 Test error= 0.283780
% lambda = 372.759372 Train error= 0.320140 Test error= 0.347791
% lambda = 1000.000000 Train error= 0.422835 Test error= 0.439387
% 
%  --FEATURES: 1000
% lambda = 0.001000 Train error= 0.210204 Test error= 0.261303
% lambda = 0.002683 Train error= 0.210230 Test error= 0.261303
% lambda = 0.007197 Train error= 0.210255 Test error= 0.261405
% lambda = 0.019307 Train error= 0.210217 Test error= 0.261507
% lambda = 0.051795 Train error= 0.210230 Test error= 0.261354
% lambda = 0.138950 Train error= 0.210217 Test error= 0.260996
% lambda = 0.372759 Train error= 0.209808 Test error= 0.260536
% lambda = 1.000000 Train error= 0.209808 Test error= 0.260383
% lambda = 2.682696 Train error= 0.209808 Test error= 0.259566
% lambda = 7.196857 Train error= 0.210690 Test error= 0.258135
% lambda = 19.306977 Train error= 0.215377 Test error= 0.256552
% lambda = 51.794747 Train error= 0.231252 Test error= 0.266360
% lambda = 138.949549 Train error= 0.270945 Test error= 0.295632
% lambda = 372.759372 Train error= 0.336156 Test error= 0.353001
% lambda = 1000.000000 Train error= 0.429234 Test error= 0.440766
% 
%  --FEATURES: 100
% lambda = 0.001000 Train error= 0.370894 Test error= 0.378595
% lambda = 0.002683 Train error= 0.370894 Test error= 0.378595
% lambda = 0.007197 Train error= 0.370894 Test error= 0.378595
% lambda = 0.019307 Train error= 0.370894 Test error= 0.378493
% lambda = 0.051795 Train error= 0.370894 Test error= 0.378493
% lambda = 0.138950 Train error= 0.370881 Test error= 0.378544
% lambda = 0.372759 Train error= 0.370868 Test error= 0.378595
% lambda = 1.000000 Train error= 0.370754 Test error= 0.378442
% lambda = 2.682696 Train error= 0.370728 Test error= 0.378697
% lambda = 7.196857 Train error= 0.371239 Test error= 0.379515
% lambda = 19.306977 Train error= 0.372478 Test error= 0.380230
% lambda = 51.794747 Train error= 0.377280 Test error= 0.384623
% lambda = 138.949549 Train error= 0.388736 Test error= 0.395453
% lambda = 372.759372 Train error= 0.417497 Test error= 0.422120
% lambda = 1000.000000 Train error= 0.470779 Test error= 0.473563
% 
%  --FEATURES: 3
% lambda = 0.001000 Train error= 0.572324 Test error= 0.573436
% lambda = 0.002683 Train error= 0.572312 Test error= 0.573436
% lambda = 0.007197 Train error= 0.572312 Test error= 0.573436
% lambda = 0.019307 Train error= 0.572312 Test error= 0.573436
% lambda = 0.051795 Train error= 0.572312 Test error= 0.573436
% lambda = 0.138950 Train error= 0.572312 Test error= 0.573436
% lambda = 0.372759 Train error= 0.572324 Test error= 0.573436
% lambda = 1.000000 Train error= 0.572350 Test error= 0.573384
% lambda = 2.682696 Train error= 0.572503 Test error= 0.573538
% lambda = 7.196857 Train error= 0.572656 Test error= 0.573589
% lambda = 19.306977 Train error= 0.573167 Test error= 0.573640
% lambda = 51.794747 Train error= 0.574049 Test error= 0.573640
% lambda = 138.949549 Train error= 0.575645 Test error= 0.575734
% lambda = 372.759372 Train error= 0.580562 Test error= 0.580179
% lambda = 1000.000000 Train error= 0.589783 Test error= 0.590549



%%
nLambda = 15;
cost13000 = [0.542273 0.521328 0.491188 0.456705 0.414559 0.368582 0.318161 0.276833 0.244138 0.227893 0.226667 0.242095 0.282299 0.348404 0.440562];
cost8000 = [0.451648 0.446437 0.433359 0.418186 0.391775 0.363525 0.326130 0.285773 0.250421 0.230192 0.225849 0.241635 0.281686 0.348301 0.440307];
cost5000 = [0.323372 0.323065 0.321430 0.319591 0.316373 0.308097 0.294100 0.275709 0.250728 0.232899 0.227842 0.242350 0.281430 0.347842 0.439847]; 
cost3000 = [0.277241 0.277395 0.277292 0.276322 0.275811 0.274432 0.271418 0.264623 0.253640 0.239489 0.233768 0.246130 0.283780 0.347791 0.439387];
cost1000 = [0.261303 0.261303 0.261405 0.261507 0.261354 0.260996 0.260536 0.260383 0.259566 0.258135 0.256552 0.266360 0.295632 0.353001 0.440766];
cost100 = [0.378595 0.378595 0.378595 0.378493 0.378493 0.378544 0.378595 0.378442 0.378697 0.379515 0.380230 0.384623 0.395453 0.422120 0.473563];
cost3= [0.573436 0.573436 0.573436 0.573436 0.573436 0.573436 0.573436 0.573384 0.573538 0.573589 0.573640 0.573640 0.575734 0.580179 0.590549];

lambdaSpace = logspace(-3,3,nLambda);
figure
semilogx(lambdaSpace, cost13000, 'r');
hold on
semilogx(lambdaSpace, cost8000, 'y');
semilogx(lambdaSpace, cost5000, 'm');
semilogx(lambdaSpace, cost3000, 'g');
semilogx(lambdaSpace, cost1000, 'c');
semilogx(lambdaSpace, cost100, 'b');
semilogx(lambdaSpace, cost3, 'k');
xlabel('\lambda') % x-axis label
ylabel('Median error') % y-axis label
legend('13000','8000','5000','3000','1000','100','3');














