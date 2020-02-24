clc, clear, close all

%% ucitavanje
t = readtable('letter-recognition.csv');
sz = size(t);
ulazi = t(:,2:sz(1,2));
strIzlazi = t(:,1);
CLASS_NUM = 26;
ulazi = table2array(ulazi);
strIzlazi = table2array(strIzlazi);
strIzlazi = cellfun(@double, strIzlazi); %pretvara u ASCII kod
izlazi = strIzlazi - 64;
izlazi = rot90(izlazi);
ulazi = rot90(ulazi);


% po klasama + histogram
K = cell(1,CLASS_NUM);
for i = 1:CLASS_NUM
    K{1,i} = ulazi(:,izlazi == i);
end

C = categorical(izlazi, [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26], {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'});
h = histogram(C, 'BarWidth', 0.5);

% % podela podataka
trainUlaz = [];
testUlaz = [];
trainIzlaz_privremeno = [];
testIzlaz_privremeno = [];

for i = 1:CLASS_NUM
    sz = size(K{1,i});
    n = sz(1,2);
    granica = ceil(0.8*n);
    trainUlaz = [ trainUlaz [K{1,i}(:, 1:granica)]];
    testUlaz = [ testUlaz [K{1,i}(:, granica+1:n)]];
    trainIzlaz_privremeno = [ trainIzlaz_privremeno  i*ones(1,granica) ];
    testIzlaz_privremeno = [ testIzlaz_privremeno i*ones(1,n-granica) ];
end

trainIzlaz = [];
testIzlaz = [];
for i = 1:length(trainIzlaz_privremeno)
    tab = zeros(1,26);
    tab(1,trainIzlaz_privremeno(1,i)) = 1;%stavlja 1 na mesto te klase, a sve ostalo su nule
    trainIzlaz = [ trainIzlaz; tab ];
end
trainIzlaz = rot90(trainIzlaz, 3);


for i = 1:length(testIzlaz_privremeno)
    tab = zeros(1,26);
    tab(1,testIzlaz_privremeno(1,i)) = 1;
    testIzlaz = [ testIzlaz; tab ];
end
testIzlaz = rot90(testIzlaz, 3);

ind = randperm(length(trainUlaz));
trainUlaz = trainUlaz(:,ind);
trainIzlaz = trainIzlaz(:,ind);

ind = randperm(length(testUlaz));
testUlaz = testUlaz(:,ind);
testIzlaz = testIzlaz(:,ind);


% struktura
Ntrain = length(trainUlaz);

acc = 0;
bestStruct = [8 5 3];
bestActivation = 'poslin';
bestRegularization = 0.01;
bestEpochNum = 1000;
bestlr = 0.1;
Xval = trainUlaz(:, [ceil(0.8*Ntrain)+1:Ntrain]);
Yval = trainIzlaz(:, [ceil(0.8*Ntrain)+1:Ntrain]);
% [12 27 77 11 3], [20 20 200 20 2], [10 171 15 200 3], [22 300 22], [10 200 10], [10 10 10], [20 17 11 17], [23 15 5 12], ]

% petlja
for structure = {[8 3] [10 10 20]}
    
    net = patternnet(structure{1});
    net.layers{length(structure{1})+1}.transferFcn = 'softmax';
    net.layers{length(structure{1})+1}.size = CLASS_NUM;
    
    for f = {'tansig', 'logsig', 'poslin'}
        
        for i = 1:length(structure{1})
            net.layers{i}.transferFcn = f{1};
        end
        
        for regularization = 0:0.1:1
            net.performParam.regularization = regularization;
            net.divideFcn = 'divideind';
            net.divideParam.trainInd = [1:ceil(0.8*Ntrain)];
            net.divideParam.testInd = [];
            net.divideParam.valInd = [ceil(0.8*Ntrain)+1:Ntrain];
            net.trainParam.max_fail = 10;
            net.trainParam.goal = 10e-8;
            net.trainParam.min_grad = 10e-8;
            net.trainParam.epochs = 200;
            net.trainParam.showWindow = true;
            
            for lr = {0.1, 0.05, 0.01, 0.005}                
                net.trainParam.lr = lr;
                [net, tr] = train(net, trainUlaz, trainIzlaz); %za balansirane

                Yval_pred = net(Xval);

                [c, cm] = confusion(Yval, Yval_pred);
                cm = cm';

                [A, F] = getPandF(cm);
                lr
                 if A > acc & not(isnan(A))
                    acc = A;
                    bestStruct = structure{1};
                    bestActivation = f{1};
                    bestRegularization = regularization;
                    bestEpochNum = tr.best_epoch;
                    bestlr = lr;
                 end
            end
        end
    end
end

% nastavak
net = patternnet(bestStruct);

for i = 1:1:length(bestStruct)
    net.layers{i}.transferFcn = bestActivation;
end
net.layers{length(bestStruct)+1}.transferFcn = 'softmax';
net.layers{length(bestStruct)+1}.size = CLASS_NUM;
net.performParam.regularization = bestRegularization;
net.divideFcn = '';
net.trainParam.max_fail = 10; %15
net.trainParam.goal = 10e-9;
net.trainParam.min_grad = 10e-9;
net.trainParam.epochs = bestEpochNum;
net.trainparam.showWindow = true;
net.trainParam.lr = bestlr;

net = train(net, trainUlaz, trainIzlaz) %za balansirane

Ntest = length(testUlaz);

% PREDIKCIJA NAD TRAIN SKUPOM
Ytrain_pred = net(trainUlaz);
figure
plotconfusion(trainIzlaz, Ytrain_pred);

[c, cm] = confusion(trainIzlaz, Ytrain_pred);
cm = cm';


% PREDIKCIJA NAD TEST SKUPOM
Ytest_pred = net(testUlaz);
figure
plotconfusion(testIzlaz, Ytest_pred);

[c, cm] = confusion(testIzlaz, Ytest_pred);
cm = cm';

acc=cm(26,26)/sum(cm(26,:))
recall=cm(26,26)/sum(cm(:,26))
