clc, clear, close all

%% Ucitavanje odbiraka
load 'dataset3.mat'
% Ucitani podaci su podeljeni tako da prvi red predstavlja x1, drugi red
% predstvalja x2, a treci red predstavlja izlaz y. x1 i x2 zajedno cine
% ulaze u NM
pod = rot90(data, 3); %rotira dataset3
%%%%%% Izdvojiti ulaze i izlaz NM
ulaz = pod(1:2,:);
izlaz = pod(3,:);

% Na osnovu izlaza, podeliti ulazne podatke na K1 (y = 0) i K2 (y = 1)
%%%%%% Podeliti odbirke na K1 i K2
K1 = ulaz(:,izlaz==0); 
K2 = ulaz(:,izlaz==1);


%% Prikaz podataka
figure, hold all
%%%%%% Prikazati ucitane podatke
scatter(K1(1,:), K1(2,:),'r.');
scatter(K2(1,:), K2(2,:),'b.');

%% Podela podataka na trening i test skup
N = length(ulaz);

%%%%%% Promesati indekse na slucajan nacin
ind = randperm(N);

%%%%%% Uzeti odbirke koji ce pripadati trening skupu
ulazTrening = ulaz(:, ind(1:0.8*N));
izlazTrening = izlaz(:,ind(1:0.8*N));

%%%%%% Uzeti odbirke koji ce pripadati test skupu
ulazTest = ulaz(:, ind(0.8*N+1:N));
izlazTest = izlaz(:,ind(0.8*N+1:N));

%% Kreiranje neuralne mreze
net1 = patternnet([4 4 4 4]);
net2 = patternnet([2 2 2]); %underFit
net3 = patternnet([50 50 50 50]); %overFit

%%%%%% Podesiti parametre neuralne mreze
net1.trainParam.epochs = 500;
net2.trainParam.epochs = 500;
net3.trainParam.epochs = 500;

net1.trainParam.goal = 0.000001;
net2.trainParam.goal = 0.000001;
net3.trainParam.goal = 0.000001;

%%%%%% Podesiti parametre zastite od preobucavanja koriscenjem val skupa
net1.divideFcn='';% ukida divide fcn
net2.divideFcn='';% ukida divide fcn
net3.divideFcn='';% ukida divide fcn

%% Treniranje i testiranje neuralne mreze
%%%%%% Obuciti NM nad trening skupom
net1 = train(net1,ulazTrening,izlazTrening);
net2 = train(net2,ulazTrening,izlazTrening);
net3 = train(net3,ulazTrening,izlazTrening);

%%%%%% izvrsiti predikciju NM nad test podacima
izlazPredTest1 = sim(net1, ulazTest);
izlazPredTest2 = sim(net2, ulazTest);
izlazPredTest3 = sim(net3, ulazTest);

%%%%%% Prikazati matricu konfuzije za trening skup
figure 
plotconfusion(izlazTest, izlazPredTest1, 'TestOpt');
figure 
plotconfusion(izlazTest, izlazPredTest2, 'TestUnder');
figure 
plotconfusion(izlazTest, izlazPredTest3, 'TestOver');

%%%%%% izvrsiti predikciju NM nad trening podacima
izlazPredTrening1 = sim(net1,ulazTrening);
izlazPredTrening2 = sim(net2,ulazTrening);
izlazPredTrening3 = sim(net3,ulazTrening);


%%%%%% Prikazati matricu konfuzije za test skup
figure 
plotconfusion(izlazTrening, izlazPredTrening1, 'TreningOpt');
figure
plotconfusion(izlazTrening, izlazPredTrening2, 'TreningUnder');
figure
plotconfusion(izlazTrening, izlazPredTrening3, 'TreningOver');

%% Granica odlucivanja
Ntest = 200;
% Formirati ulazni vektor za testiranje
xTest = linspace(-1, 1, Ntest);
yTest = linspace(-1, 1, Ntest);
ulazTestGO = [];
for i = xTest
    ulazTestGO = [ulazTestGO [i*ones(size(yTest)); yTest]];
end

%%%%%% Testirati obucen perceptron za formiranu mrezu podataka
izlazTest1 = sim(net1,ulazTestGO);
izlazTest2 = sim(net2,ulazTestGO);
izlazTest3 = sim(net3,ulazTestGO);

%%%%%% Podeliti prediktovane izlaze u K1p, K2p i Kn u zavisnoti od praga
%%%%%% odlucivanja
K1p1 =  ulazTestGO(:,izlazTest1<0.35);
K2p1 = ulazTestGO(:,izlazTest1>0.65); 
Kn1 = ulazTestGO(:,izlazTest1>0.35 & izlazTest1<0.65 );

K1p2 =  ulazTestGO(:,izlazTest2<0.35);
K2p2 = ulazTestGO(:,izlazTest2>0.65); 
Kn2 = ulazTestGO(:,izlazTest2>0.35 & izlazTest2<0.65 );
 
K1p3 =  ulazTestGO(:,izlazTest3<0.35);
K2p3 = ulazTestGO(:,izlazTest3>0.65); 
Kn3 = ulazTestGO(:,izlazTest3>0.35 & izlazTest3<0.65 );

%%%%%% Prikazati granicu odlucivanja
figure
plot(K1p1(1,:),K1p1(2,:),'r.',K2p1(1,:),K2p1(2,:),'b.', Kn1(1,:),Kn1(2,:), 'g.');

figure
plot(K1p2(1,:),K1p2(2,:),'r.',K2p2(1,:),K2p2(2,:),'b.', Kn2(1,:),Kn2(2,:), 'g.');
 
figure
plot(K1p3(1,:),K1p3(2,:),'r.',K2p3(1,:),K2p3(2,:),'b.', Kn3(1,:),Kn3(2,:), 'g.');
