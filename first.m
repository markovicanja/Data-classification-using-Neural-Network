clc, clear, close all

%% Generisanje funkcije koju treba "fitovati"
N = 1000;

%%%%%% Ulaz je vektor koji predtavlja vremensku osu
x = linspace(0,1.5,N);


%%%%%% Definisati funkciju h
h = 2*sin(2*3.14*15*x)+3*sin(2*3.14*9*x);

%%%%%% Kreirati izlaz NM tako sto se na funkciju f dodaje sum
y = h + randn(1,N)*0.4;

%% Prikaz podataka
figure
hold all
%%%%%% Prikazati ucitane podatke
plot(x,h,'r');
plot(x,y,'b');

%% Podela podataka na trening i test skup
%%%%%% Promesati indekse na slucajan nacin
ind = randperm(N);
%ind je vektor slucajne permutacije brojeva od 1-1000

%%%%%% Uzeti odbirke koji ce pripadati trening skupu 
xTrening = x(:,ind(1:0.8*N));
yTrening = y(:,ind(1:0.8*N));

%%%%%% Uzeti odbirke koji ce pripadati test skupu
xTest = x(:,ind(0.8*N+1:N));
yTest = y(:,ind(0.8*N+1:N));

%% Kreiranje neuralne mreze 
net = fitnet([10 8 10]);

%%%%%% Podesiti parametre neuralne mreze
net.performFcn= 'mse'; %kriterijumska fja
net.divideFcn=''; %ukida validaciju, da mi on ne deli ulaz na trening-validaciju
net.trainParam.epochs= 500;  %% oko 200-500
net.trainParam.goal= 0.000001; %max greska, oko 0.0001  - 0.000001

%%%%%% Obuciti NM nad trening skupom
net = train(net,xTrening,yTrening);

%% Testiranje NM i prikaz rezultata
%%%%%% Izvrsiti klasifikaciju nad test podacima
yPred = sim(net,xTest);

%%%%%% Izvrsiti klasifikaciju nad svim podacima
hPred = sim(net,x);

figure, hold all
%%%%%% Prikazati na istom grafiku funckiju f i rezultat predikcije na celom skupu podataka
plot(x, h, 'b', x, hPred, 'r');